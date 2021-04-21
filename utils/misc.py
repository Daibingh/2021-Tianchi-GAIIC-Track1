from io import StringIO
import pandas as pd

def cvt_fmtb(file):
    with open(file, 'r') as f:
        t = f.read()
    t = t.replace('|,|', ',')
    t = t.replace(' ', ',')
    df = pd.read_csv(StringIO(t), header=None, index_col=0)
    return df

def cvt_fmt(df, file):
    df = df.apply(lambda t: ' '.join(["{}".format(t) for t in t.tolist()]), axis=1)
    df.to_csv(file, index=True, header=False)
    with open(file) as f:
        t = f.read().replace(',', '|,|')
    with open(file, 'w') as f:
        f.write(t)

def scoring(loss):
    return 1 - 2*loss