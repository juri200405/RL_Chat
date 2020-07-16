import unicodedata
import re

def normalize_string(st):
    '''
    文st中に含まれる文字を正規化
    '''
    st = ''.join(c for c in unicodedata.normalize('NFKC', st) if unicodedata.category(c) != 'Mn')
    st = "\n".join([re.sub(r"\s+", r" ", s).strip() for s in re.split(r"\n", st) if s != ""])
    return st

if __name__ == '__main__':
    print(normalize_string("hello   world"))
    st = '''
    本日は




    晴天　なり
    '''
    print(st)
    print(normalize_string(st))
