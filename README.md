# movie recommendation system

![image](https://github.com/user-attachments/assets/fad3f061-f70a-450f-b930-22a7d31aeb3d)



## 초기설정

### Requirements
- python
- Flask
- MySQL
- html

## 1. 입력 받은 dataid를 통해 사용자의 직업과 같은 사람들이 가장 좋아하는 무비 10개  추천
```
 sql0 = "SELECT occupation FROM user WHERE userid="+str(num1)+";"
cur.execute(sql0)
rows = cur.fetchall()[0][0]
----rows에 데이터베이스를 통한 직업 데이터 받기.

sql2 = "select movieTitle,avg(ratingScore) as ratingScore 
from user natural join ratings natural join movie 
----user, movie, rating natural join을 한다.
where occupation='"+str(rows)+
----아까 받은 rows의 직업 데이터를 이용하여 해당 유저의 직업을 가진 데이터로 한정
"' group by movieid 
----movie id로 묶고
order by ratingScore desc,movieId desc limit 10;"
----평점순 10개만 출력

cur.execute(sql2)
result=cur.fetchall()
----result에 저장
```


## 2.입력 받은 dataid를 통해 사용자의 나이대(10’s, 20’s, 30’s, 40’s, 50’s, …)와 같은 사람들의 가장 좋아하는 무비 10개  추천
```
sql3="SELECT age FROM user WHERE userid="+str(num1)+";"
cur.execute(sql3)
age = int(cur.fetchall()[0][0])
----age에 데이터베이스를 통한 나이 데이터 받기.

min_age=math.floor(age/10)*10)
----1의 자리 버림을 통한 (10’s, 20’s, 30’s, 40’s, 50’s, …)결정
max_age=min_age+10
----+10 (ex)10-20같은 범위 결정

sql4="select movieTitle,avg(ratingScore) as ratingScore 
from user natural join ratings natural join movie 
----user, movie, rating natural join을 한다.
where "+str(min_age)+" <=age and age < "+str(max_age)+" 
----자신의 나이대에 속하는 나이대의 데이터만 남긴다
"' group by movieid 
----movie id로 묶고
order by ratingScore desc,movieId desc limit 10;"
----평점순 10개만 출력
cur.execute(sql4)
result2=cur.fetchall()
----result2에 저장
```

## 3.knn 추천 알고리즘(K=40, 피어슨 상관계수, 유저 기반)
 ![image](https://github.com/user-attachments/assets/8815e96b-4ec5-489d-8366-db78f0ff6b9f)
```
----다른사람과 유사도 검사를 통해 영화 추천을 받는 유저(A)를 userid로 넣는다
def cosim(userid, dataframe):
    movies = []
    for i in dataframe.loc[userid, :].index:
        if math.isnan(dataframe.loc[userid, i]) == False:
            movies.append(i)
----movies에 유사도 검사유저가 본 영화 저장
    U_df = pd.DataFrame(dataframe.loc[userid, movies]).T
----U_df 유저가 유저가 본 영화에 대한 평점을 데이터 프레임에 저장
    other_df = dataframe.loc[:, movies].drop(userid, axis=0)
----other_df에는 다른 모든 유저가 유저가 본 영화에 대한 평점을 데이터 프레임에 저장
    sim_dict = {}
    for user in other_df.index:
----다른 유저 집합중 한 유저 선택(B)
        u_list = []
        df_list = []
        ch1 = 0
        pa1 = 0
        pa2 = 0
        sm = [m for m in U_df.columns if math.isnan(other_df.loc[user, m]) ==
False]
----다른 유저(B)가 유저(A)가 본 영화를 전부 보지 못 했기 때문에 NaN이 아닌 영화만 sm에 저장
        for i in U_df.loc[userid, sm].values:
            u_list.append(i)
---- sm에 있는 영화(유저(A)가 본 영화)에 대한 유저(A)의 평가를 u_list에 순서대로 저장
        for i in other_df.loc[user, sm].values:
            df_list.append(i)
---- sm에 있는 영화(유저(A)가 본 영화)에 대한 다른 유저(B)의 평가를 df_list에 순서대로 저장
        for i in range(len(sm)):
            ch1 += (u_list[i] * df_list[i])
            pa1 += u_list[i] * u_list[i]
            pa2 += df_list[i] * df_list[i]
        ch2 = sum(u_list) * sum(df_list)
        pa3 = (pa1 - sum(u_list) * sum(u_list) / (len(sm) + 0.000000001)) * (
                    pa2 - sum(df_list) * sum(df_list) / (len(sm) + 0.000000001))
        result = (ch1 - ch2 / (len(sm) + 0.000000001)) / ((pow(pa3, 1 / 2) + 0.000000001))
```
![image](https://github.com/user-attachments/assets/a97bdd6c-42c0-415a-bd92-01c99a2f2e47)

```
---- 피어슨 상관 계수(아래 우측 이미지)를 좌측 같이 변형하여 계산하는 과정, 분모가 0이 되는 것을 고려하여 0.000000001추가
        sim_dict[user] = result
----sim_dict에 다른 유저(B)의 userid와 해당 유저와 유저(A)와의 상관 계수를 딕셔너리로 저장
    ''''''
    sim_mat = sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True)[:40]
----sim_mat에는 모든 다른 유저와의 상관 계수 중 k=40임으로 상위 40명을 저장
```
![image](https://github.com/user-attachments/assets/3b983ac6-b939-441f-9772-334537abb001)

```
----knn계산과정----
    recommend_list = list(set(dataframe.columns) - set(U_df.columns))
---- 유저(A)가 안본 영화 리스트를 recommend_list 저장 
    others_k = [i[0] for i in sim_mat]
---- 유사도 높은 40명의 userid를 저장
    recommender = dict()
    for movie in recommend_list:
---- 영화 하나를 결정
        rating = []
        sim = []
        for person in others_k:
            if math.isnan(dataframe.loc[person, movie]) == False:
                rating.append(dataframe.loc[person, movie])
                sim.append(sim_dict[person])
---- 다른 유저 한 명씩 불러오고 위에서 결정된 영화를 해당 유저가 봤다면 영화 평점과 유사도 데이터를 순서대로 rating,sim에 저장

        aa = len(rating)
        use_sum = 0
        for i in range(aa):
            use_sum += sim[i] * rating[i]
        pred = use_sum / (sum(sim) + 0.0000001)
        recommender[movie] = pred
----아래 계산대로 유저(A)가 안 본 영화의 예상 평점을 recommender dict에 저장
    return sorted(recommender.items(), key=operator.itemgetter(1), reverse=True)[:10]
---- recommender dict에서 평점 높은 10개 영화 출력
```



