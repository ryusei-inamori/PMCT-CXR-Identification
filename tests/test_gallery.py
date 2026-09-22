import pandas as pd
from pmct_cxr_id.gallery import select_category

def test_gallery_selection():
    q=pd.DataFrame([{"query_id":"q1","identity_id":"A","image_path":"q.png","pmct_date":pd.Timestamp("2020-01-10")}])
    g=pd.DataFrame([
        {"identity_id":"A","image_path":"old.png","exam_date":pd.Timestamp("2010-01-01"),"is_distractor":False},
        {"identity_id":"A","image_path":"near.png","exam_date":pd.Timestamp("2020-01-01"),"is_distractor":False},
        {"identity_id":"D","image_path":"d.png","exam_date":pd.NaT,"is_distractor":True},
    ])
    assert "old.png" in set(select_category(g,q,"oldest").image_path)
    assert "near.png" in set(select_category(g,q,"nearest").image_path)
    assert len(select_category(g,q,"all")) == 3
