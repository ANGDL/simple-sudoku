import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def cv_show_image(name, src):
    cv2.imshow(name, src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class ImageToArray:
    def __init__(self, template):
        self.template = template
        self.ref_imgs = self.get_reference(template)
        assert len(self.ref_imgs) == 9

    def sort_coutours(self, cnts, ax=0, reverse=False):
        assert ax < 4
        bboxes = [cv2.boundingRect(c) for c in cnts]
        cnts, bboxes = zip(*sorted(zip(cnts, bboxes), key=lambda b: b[1][ax], reverse=reverse))
        return cnts, np.array(bboxes)

    def get_reference(self, template):
        img = cv2.imread(template)
        ref_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ref_img = cv2.threshold(ref_img, 10, 255, cv2.THRESH_BINARY_INV)[1]
        cnts, _ = cv2.findContours(ref_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts, bboxes = self.sort_coutours(cnts)
        ref_imgs = [ref_img[y:y + h, x:x + w] for x, y, w, h in bboxes]

        # img_cp1 = img.copy()
        # for i in range(len(bboxes)):
        #     p1 = (bboxes[i, 0], bboxes[i, 1])
        #     p2 = (bboxes[i, 0] + bboxes[i, 2], bboxes[i, 1] + bboxes[i, 3])
        #     cv2.rectangle(img_cp1, p1, p2, (255, 0, 0), 2)
        #
        # cv_show_image("", img_cp1)

        return ref_imgs

    def __call__(self, sudoku_img):
        # 读取图片
        sudoku = cv2.imread("sudoku1.jpg")
        # 缩放
        scale = 0.5
        w = int(sudoku.shape[1] * scale)
        h = int(sudoku.shape[0] * scale)
        sudoku = cv2.resize(sudoku, (w, h))
        # 灰度图，边缘检测
        sudoku_gray = cv2.cvtColor(sudoku, cv2.COLOR_RGB2GRAY)
        sudoku_edge = cv2.Canny(sudoku_gray, 100, 200)
        _, sudoku_bin = cv2.threshold(sudoku_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # 加深边缘信息
        kernel = np.ones((3, 3), np.uint8)
        sudoku_edge = cv2.dilate(sudoku_edge, kernel, iterations=1)
        # 查找数字区域
        contours, hierarchy = cv2.findContours(sudoku_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        locs = []
        areas = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if 1.1 > w / h > 0.9 or 1.1 > h / w > 0.9:
                locs.append([x, y, x + w, y + h])
                areas.append(cv2.contourArea(cnt))

        idx = np.argmax(areas)
        box_loc = locs[idx]
        x1, y1, x2, y2 = box_loc
        box = sudoku_bin[y1:y2, x1:x2]
        # plt.imshow(box, cmap='gray')
        # plt.show()
        s = max(box.shape[0] - box.shape[0] % 9, box.shape[1] - box.shape[1] % 9)
        box = cv2.resize(box, (s, s))
        # 拆分数字
        gride_w = 9
        gride_h = 9
        stride = int(box.shape[0] / gride_w)

        gride_digits = []

        for x in range(gride_w):
            for y in range(gride_h):
                digit = box[x * stride + 5: (x + 1) * stride - 5, y * stride + 5: (y + 1) * stride - 5]
                gride_digits.append(digit)

        # 模板匹配
        output = []
        for digit_img in gride_digits:
            cnts, _ = cv2.findContours(digit_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) == 0:
                output.append('.')
                continue
            x, y, w, h = cv2.boundingRect(cnts[0])
            roi = digit_img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))
            roi = cv2.threshold(roi, 10, 255, cv2.THRESH_BINARY)[1]
            #     cv_show_image("", roi)
            scores = []
            for ref_digit_img in self.ref_imgs:
                ref_digit_img = cv2.resize(ref_digit_img, (57, 88))
                ref_digit_img = cv2.threshold(ref_digit_img, 10, 255, cv2.THRESH_BINARY)[1]
                #         cv_show_image("", ref_digit_img)
                result = cv2.matchTemplate(roi, ref_digit_img, cv2.TM_CCORR_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
                scores.append(score)

            output.append(str(np.argmax(scores) + 1))

        return np.array(output).reshape(9, 9)


class SolutionSudoku:
    def __init__(self):
        self.is_solved = False

    def solve(self, board):
        row = [set(range(1, 10)) for _ in range(9)]  # 行剩余可用数字
        col = [set(range(1, 10)) for _ in range(9)]  # 列剩余可用数字
        block = [set(range(1, 10)) for _ in range(9)]  # 块剩余可用数字

        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':  # 更新可用数字
                    val = int(board[i][j])
                    row[i].remove(val)
                    col[j].remove(val)
                    block[(i // 3) * 3 + j // 3].remove(val)

        def backtrack(it=0):
            if it > 80:
                self.is_solved = True
                return

            i = it // 9
            j = it % 9
            b = (i // 3) * 3 + j // 3

            if board[i][j] == '.':

                for n in range(1, 11):
                    if (n in row[i]) and (n in col[j]) and (n in block[b]):
                        row[i].remove(n)
                        col[j].remove(n)
                        block[b].remove(n)
                        board[i][j] = str(n)

                        backtrack(it + 1)
                        if self.is_solved:
                            return
                        row[i].add(n)
                        col[j].add(n)
                        block[b].add(n)
                        board[i][j] = '.'
            else:
                backtrack(it + 1)

        backtrack()

    def __call__(self, template, sudoku_img):
        self.is_solved = False
        board = ImageToArray(template)(sudoku_img)
        self.solve(board)
        return board


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--template", default='123456789.jpg',
                    help="path to input image")
    parser.add_argument("-t", "--image", default='sudoku1.jpg',
                    help="path to template image")

    args = parser.parse_args()

    template = args.template
    img = args.image

    solution = SolutionSudoku()
    out = solution(template, img)

    print(out)
