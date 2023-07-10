import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils_common.utils import BoundaryUtil


def _main():
    check_boundary_recall()


def _procedures():
    pass


def check_boundary_recall():
    s = [
        '雪在山上发白就在今夜',
        '不被望见的脚印',
        '啊一个孤立的国界',
        '且看我是女王也',
        '风内在哀嚎像这缠绵的风暴',
        '无法挽留天知我努力过',
        '别进来别让人看到',
        '乖乖女孩你永远都要好',
        '关心不让他们感受',
        '嗯现在都',
        '山上的白雪染白了今夜',
        '一个个孤立的国界',
    ]
    b = [
        '1000010000',
        '0100000',
        '10001000',
        '0101000',
        '000010000000',
        '0000100000',
        '00010000',
        '0001001000',
        '00010000',
        '1000',
        '1000010000',
        '10001000',
    ]
    metric = BoundaryRecall()
    print(metric.boundary_recall_batch(s, b))


class BoundaryRecall:
    def __init__(self):
        self.boundary_util = BoundaryUtil()

    def boundary_recall_batch(self, out_sentence, tgt_boundary, verbose=False):
        '''
        Compute boundary recall score in batch
        out_sentence: a list of output sentence in Chinese
        tgt_boundary: a list of boundary target, each target is a binary string.
        '''
        assert len(out_sentence) == len(tgt_boundary)
        assert isinstance(out_sentence[0], str)
        assert isinstance(tgt_boundary[0], str)

        cnt, cnt_tgt = 0, 0
        for out_sentence_s, tgt_boundary_s in zip(out_sentence, tgt_boundary):
            out_boundary = self.boundary_util.get_all_boundary(out_sentence_s)
            if verbose == True:
                print(out_boundary, tgt_boundary_s)
            t1, t2 = self.get_true_positive(out_boundary, tgt_boundary_s)
            cnt += t1
            cnt_tgt += t2

        print('True positive & Total: ', cnt, cnt_tgt)
        return cnt / cnt_tgt

    def get_true_positive(self, out_boundary_s, tgt_boundary_s):
        '''
        Count
        - the number of true positive in one sentence
        - the number of target in ground truth
        '''
        cnt = 0
        cnt_tgt = 0
        for i in range(len(tgt_boundary_s)):
            if tgt_boundary_s[i] == '1':
                cnt_tgt += 1
                if i <= len(out_boundary_s) - 1 and out_boundary_s[i] == '1':
                    cnt += 1
        return cnt, cnt_tgt


if __name__ == '__main__':
    _main()
