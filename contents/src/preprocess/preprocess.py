import unicodedata
import re

def normalize_string(s):
    '''
    文s中に含まれる文字を正規化し、連続する改行、空白をそれぞれ一つのものに置き換える。
    '''
    s = ''.join(c for c in unicodedate.normalize('NFKC', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r"(\s)+", r'\1', s).strip()
    return s

if __name__ == '__main__':
    print(normalize("hello   world"))
