import datetime

today = datetime.date.today()

class Family:
    def __init__(self,
        sex,  # 余命や疾病リスク計算のため社会的性別(gender)ではなく生物学的性別(sex)を入力
        birth_year,
        birth_month=1,
        relationship="本人",  # ["本人", "配偶者", "子ども", "親"]
        is_dependent=False,
        is_disabled=False
    ):
        if isinstance(sex, int):
            if sex == 0:
                self.sex = "女"
            elif sex == 1:
                self.sex = "男"
            else:
                self.sex = "不明"
        elif isinstance(sex, str):
            sex = sex.lower()
            if sex in ["女", "女性", "female", "f", "woman"]:
                self.sex = "女"
            elif sex == ["男", "男性", "male", "m", "man"]:
                self.sex = "男"
            else:
                self.sex = "不明"
        else:
            self.sex = "不明"

        if isinstance(birth_year, str):
            try:
                birth_year = int(birth_year)
            except:
                self.birth_year = 1990

        if isinstance(birth_year, int):
            if (birth_year >= 1900) & (birth_year <= today.year):
                self.birth_year = birth_year
            else:
                self.birth_year = 1990
        else:
            self.birth_year = 1990
            
        if isinstance(birth_month, int):
            if (birth_month >= 1) & (birth_month <= 12):
                self.birth_month = birth_month
            else:
                self.birth_month = 1
        else:
            self.birth_month = 1

        self.birthday = datetime.date(birth_year, birth_month, 1)
        self.age = today - self.birthday


