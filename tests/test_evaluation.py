import pandas as pd
from pmct_cxr_id.evaluation import exact_mcnemar

def test_mcnemar():
    a=pd.DataFrame({"query_id":["1","2"],"identity_id":["A","B"],"top1_correct":[0,1]})
    b=pd.DataFrame({"query_id":["1","2"],"identity_id":["A","B"],"top1_correct":[1,1]})
    out=exact_mcnemar(a,b,ks=(1,))
    assert out.loc[0,"discordant"] == 1
