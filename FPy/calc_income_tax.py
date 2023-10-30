import pandas as pd

class TaxPlanning:
    """
    https://www.nta.go.jp/publication/pamph/koho/kurashi/html/01_1.htm
    """
    def __init__(
        self,
        interest_income=0,
        dividend_income=0,
        #realestate_income=0,
        business_income=0,
        salary=0,　　# 給与収入
        retirement_allowance=0,  # 退職金
        #forestry_income=0,
        #shortterm_capitalgain=0,
        #longterm_capitalgain=0,
        #realestate_shortterm_capitalgain=0,
        #realestate_longterm_capitalgain=0,
        temporary_income_before_cost=0,
        public_pension=0  # 公的年金
        #miscellaneous_income_excluding_pension=0
    ):
        self.interest_income = interest_income
        self.dividend_income = dividend_income
        #self.realestate_income = realestate_income
        self.business_income = business_income
        self.salary = salary
        self.retirement_allowance = retirement_allowance
        #self.forestry_income = forestry_income
        #self.shortterm_capitalgain = shortterm_capitalgain
        #self.longterm_capitalgain = longterm_capitalgain
        #self.temporary_income_before_cost = temporary_income_before_cost
        self.public_pension = public_pension
        #self.miscellaneous_income_excluding_pension = miscellaneous_income_excluding_pension
        self.basic_exemption = 380000


    # 給与所得
    def calc_employment_income(self):
        """
        給与収入から給与所得金額を計算
        """
        if self.salary <= 1625000:
            self.employment_income_deduction = 650000
        elif self.salary <= 1800000:
            self.employment_income_deduction = self.salary * 0.4
        elif self.salary <= 3600000:
            self.employment_income_deduction = self.salary * 0.3 + 180000
        elif self.salary <= 6600000:
            self.employment_income_deduction = self.salary * 0.2 + 540000
        elif self.salary <= 10000000:
            self.employment_income_deduction = self.salary * 0.1 + 1200000
        else:
            self.employment_income_deduction = 2200000
        self.employment_income = self.salary - self.employment_income_deduction  # 給与所得
        return self.employment_income

    # 退職所得
    def calc_retirement_income(self, service_year=20):
        if service_year <= 20:
            self.retirement_income_deduction = max(400000 * service_year, 800000)
        else:
            self.retirement_income_deduction = 8000000 + 700000 * (service_year - 20)
        self.retirement_income = (self.retirement_allowance - self.retirement_income_deduction) / 2
        return self.retirement_income

    # 一時所得
    def calc_temporary_income(self, cost_for_temporary_income=0):
        self.temporary_income = max(temporary_income_before_cost - cost_for_temporary_income - 500000, 0)
        return self.temporary_income

    # 雑所得(年金のみと仮定)
    def calc_miscellaneous_income(self, age=60, private_pension=0, total_private_pension=0, private_pension_premium=0):
        if age >= 65:
            if self.public_pension < 3300000:
                self.pension_deduction = 1200000
            elif self.public_pension < 4100000:
                self.pension_deduction = self.public_pension * 0.25 + 375000
            elif self.public_pension < 7700000:
                self.pension_deduction = self.public_pension * 0.15 + 785000
            else:
                self.pension_deduction = self.public_pension * 0.05 + 1555000
        else:
            if self.public_pension < 1300000:
                self.pension_deduction = 700000
            elif self.public_pension < 4100000:
                self.pension_deduction = self.public_pension * 0.25 + 375000
            elif self.public_pension < 7700000:
                self.pension_deduction = self.public_pension * 0.15 + 785000
            else:
                self.pension_deduction = self.public_pension * 0.05 + 1555000
        self.miscellaneous_income = (self.public_pension - self.pension_deduction) + (private_pension - total_private_pension / private_pension_premium)
        return self.miscellaneous_income
    
    def totalize_profit_loss(self):
        """
        損益通算し総所得を計算
        """
        total1 = self.interest_income + self.dividend_income + self.realestate_income + self.business_income + self.salary_income + self.miscellaneous_income
        total2 = self.shortterm_capitalgai + self.longterm_capitalgain + self.miscellaneous_income
        total3 = total1 + total2
        self.gross_income = total3 + self.forestry_income + self.retirement_income
        return self.gross_income


    def calc_medical_expenses_deduction(self, medical_expenses=0, highcost_medicalcare=0):
        """
        医療費控除
        """
        self.medical_expenses_deduction = max(medical_expenses - highcost_medicalcare - min(self.gross_income, 100000), 2000000)
        return self.medical_expenses_deduction

    def calc_socialinsurance_deduction(self, socialinsurance_premium=0):
        """
        社会保険料控除
        """
        self.socialinsurance_deduction = socialinsurance_premium
        return self.socialinsurance_deduction

    def calc_smallmutual_deduction(self, smallmutual_premium=0):
        """
        小規模企業共済等掛金控除
        """
        self.smallmutual_deduction = smallmutual_premium
        return self.smallmutual_deduction

    def calc_generallifeinsurance_deduction(self, annual_net_premium=0):
        """
        一般生命保険料控除
        """
        if annual_net_premium <= 20000:
            self.generallifeinsurance_deduction = annual_net_premium
        elif annual_net_premium <= 40000:
            self.generallifeinsurance_deduction = annual_net_premium / 2 + 10000
        elif annual_net_premium <= 80000:
            self.generallifeinsurance_deduction = annual_net_premium / 4 + 20000
        else:
            self.generallifeinsurance_deduction = 40000
        return self.generallifeinsurance_deduction

    def calc_individualannuity_deduction(self, annual_net_premium=0):
        """
        個人年金保険料控除
        """
        if annual_net_premium <= 20000:
            self.individualannuity_deduction = annual_net_premium
        elif annual_net_premium <= 40000:
            self.individualannuity_deduction = annual_net_premium / 2 + 10000
        elif annual_net_premium <= 80000:
            self.individualannuity_deduction = annual_net_premium / 4 + 20000
        else:
            self.individualannuity_deduction = 40000
        return self.individualannuity_deduction

    def calc_medicallongtermcareinsurance_deduction(self, annual_net_premium=0):
        """
        介護医療保険料控除
        """
        if annual_net_premium <= 20000:
            self.medicallongtermcareinsurance_deduction = annual_net_premium
        elif annual_net_premium <= 40000:
            self.medicallongtermcareinsurance_deduction = annual_net_premium / 2 + 10000
        elif annual_net_premium <= 80000:
            self.medicallongtermcareinsurance_deduction = annual_net_premium / 4 + 20000
        else:
            self.medicallongtermcareinsurance_deduction = 40000
        return self.medicallongtermcareinsurance_deduction

    def calc_lifeinsurance_deduction(self):
        """
        生命保険料控除
        """
        self.lifeinsurance_deduction = self.generallifeinsurance_deduction + self.individualannuity_deduction + self.medicallongtermcareinsurance_deduction
        return self.lifeinsurance_deduction

    def calc_earthquakeinsurance_deduction(self, premium):
        """
        地震保険料控除
        """
        self.earthquakeinsurance_deduction = min(premium, 50000)
        return self.earthquakeinsurance_deduction

    def calc_donation_deduction(self):
        """
        寄付金控除
        """
        self.donation_deduction = 0
        return self.donation_deduction

    def calc_disabledperson_deduction(self, n_disabledperson=0, n_specialdisabledperson=0, n_specialdisabledperson_livingwith=0):
        """
        障害者控除
        """
        self.disabledperson_deduction = n_disabledperson * 270000 + n_specialdisabledperson * 400000 + n_specialdisabledperson_livingwith * 750000
        return self.disabledperson_deduction

    def calc_widow_deduction(self, is_widow=False):
        """
        寡婦・寡夫控除
        """
        if is_widow:
            self.widow_deduction = 270000
        else:
            self.widow_deduction = 0
        return self.widow_deduction

    def calc_workingstudent_deduction(self, is_workingstudent=False):
        """
        勤労学生控除
        """
        if is_workingstudent & (self.gross_income <= 650000):
            self.workingstudent_deduction = 270000
        else:
            self.workingstudent_deduction = 0
        return self.workingstudent_deduction

    def calc_spouse_deduction(self, is_spouse=False, spouse_salary=0):
        """
        配偶者控除
        """
        if is_spouse & (spouse_salary <= 1030000):
            self.spouse_deduction = 380000
        else:
            self.spouse_deduction = 0
        return self.spouse_deduction
        
    def calc_dependents_deduction(self):
        """
        扶養控除
        """
        self.dependents_deduction = 0
        return self.dependents_deduction
    
    def calc_income_deduction(self):
        """
        所得控除
        """
        self.income_deduction = self.basic_exemption + self.medical_expenses_deduction + self.socialinsurance_deduction + self.smallmutual_deduction \
            + self.lifeinsurance_deduction + self.earthquakeinsurance_deduction + self.donation_deduction \
            + self.disabledperson_deduction + self.widow_deduction + self.workingstudent_deduction \
            + self.spouse_deduction + self.dependents_deduction
        return self.income_deduction

    def calc_taxable_income():
        """
        課税所得
        """
        self.taxable_income = self.gross_income - self.income_deduction
        return self.taxable_income
        
    def calc_calculated_tax(self):
        """
        算出税額
        """
        if self.taxable_income <= 1950000:
            self.tax_rate = 0.05
            self.tax_deduction = 0
        elif self.taxable_income <= 3300000:
            self.tax_rate = 0.1
            self.tax_deduction = 97500
        elif self.taxable_income <= 6950000:
            self.tax_rate = 0.2
            self.tax_deduction = 427500
        elif self.taxable_income <= 9000000:
            self.tax_rate = 0.23
            self.tax_deduction = 636000
        elif self.taxable_income <= 18000000:
            self.tax_rate = 0.33
            self.tax_deduction = 1536000
        elif self.taxable_income <= 40000000:
            self.tax_rate = 0.4
            self.tax_deduction = 2796000
        else:
            self.tax_rate = 0.45
            self.tax_deduction = 4796000
 
        self.calculated_tax = self.taxable_income * self.tax_rate - self.tax_deduction
        return self.calculated_tax

    def calc_dividend_deduction(self):
        """
        配当控除
        """
        if self.dividend_income <= 10000000:
            self.dividend_deduction = self.dividend_income * 0.1
        else:
            self.dividend_deduction = self.dividend_income * 0.05
        return self.dividend_deduction
    
    def calc_tax_credit(self, housing_loan_deduction=0):
        """
        税額控除
        """        
        self.housing_loan_deduction = housing_loan_deduction
        self.tax_credit = self.dividend_deduction + self.housing_loan_deduction
        return self.tax_credit
    
    def calc_income_tax(self):
        """
        所得税
        """
        self.income_tax = self.calculated_tax - self.tax_credit
        return self.income_tax
    
    def calc_withholding_tax():
        """"
        源泉徴収、納付税額
        """
        return None