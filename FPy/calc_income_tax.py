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
        salary=0,  # 給与収入
        retirement_allowance=0,  # 退職金
        #forestry_income=0,
        #shortterm_capitalgain=0,
        #longterm_capitalgain=0,
        #realestate_shortterm_capitalgain=0,
        #realestate_longterm_capitalgain=0,
        temporary_income_before_cost=0,
        public_pension=0,  # 公的年金
        #miscellaneous_income_excluding_pension=0
        families=None
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
        self.temporary_income_before_cost = temporary_income_before_cost
        self.public_pension = public_pension
        #self.miscellaneous_income_excluding_pension = miscellaneous_income_excluding_pension
        self.basic_exemption = 380000
        self.families = families

        self.income = self.salary + self.retirement_allowance + self.temporary_income_before_cost + self.public_pension \
            + self.business_income + self.interest_income + self.dividend_income
    
    # 給与所得
    def calc_employment_income(self):
        """
        給与収入から給与所得金額を計算
        """
        # 給与所得控除
        if self.salary <= 550000:
            self.employment_income_deduction = self.salary
        elif self.salary <= 1625000:
            self.employment_income_deduction = 550000
        elif self.salary <= 1800000:
            self.employment_income_deduction = self.salary * 0.4 - 100000
        elif self.salary <= 3600000:
            self.employment_income_deduction = self.salary * 0.3 + 80000
        elif self.salary <= 6600000:
            self.employment_income_deduction = self.salary * 0.2 + 440000
        elif self.salary <= 8500000:
            self.employment_income_deduction = self.salary * 0.1 + 1100000
        else:
            self.employment_income_deduction = 1950000

        # 所得金額調整控除
        if (self.salary > 8500000) & isinstance(self.families, collections.abc.Sequence):
            cond = False
            for family_menber in self.families:
                cond |= (family_menber.relationship == "本人") & (family_menber.is_disabled)
                cond |= family_menber.is_dependent & (self.age < 23)
                cond |= family_menber.is_dependent & (family_menber.is_disabled)
            if cond:
                employment_income_adjustment_deduction = (min(self.salary, 10000000) - 8500000) * 0.1
            else:
                employment_income_adjustment_deduction = 0
        else:
            employment_income_adjustment_deduction = 0

        if self.public_pension > 100000:
            employment_income_adjustment_deduction += 100000

        self.employment_income_adjustment_deduction = employment_income_adjustment_deduction
                
        self.employment_income = self.salary - self.employment_income_deduction  # 給与所得
        
        return self.employment_income

    # 退職所得
    def calc_retirement_income(self, service_year=20):
        if service_year <= 20:
            self.retirement_income_deduction = min(max(400000 * service_year, 800000), self.retirement_allowance)
        else:
            self.retirement_income_deduction = min(8000000 + 700000 * (service_year - 20), self.retirement_allowance)

        retirement_allowance_after_deduction = self.retirement_allowance - self.retirement_income_deduction

        if (service_year <= 5) & (retirement_allowance_after_deduction > 3000000):
            self.retirement_income = retirement_allowance_after_deduction + 1500000
        else:
            self.retirement_income = retirement_allowance_after_deduction / 2
        return self.retirement_income

    # 一時所得
    def calc_temporary_income(self, cost_for_temporary_income=0):
        self.temporary_income = max(self.temporary_income_before_cost - cost_for_temporary_income - 500000, 0)
        return self.temporary_income

    # 雑所得(年金のみと仮定)
    def calc_miscellaneous_income(self, age=60, private_pension=0, total_private_pension=0, private_pension_premium=0):
        if isinstance(self.families, collections.abc.Sequence):
            for family_menber in self.families:
                if family_menber.relationship == "本人":
                    age = family_menber.age
                    pass
                
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

        private_pension = private_pension - total_private_pension / private_pension_premium if private_pension_premium > 0 else 0
        self.miscellaneous_income = (self.public_pension - self.pension_deduction) + private_pension
        return self.miscellaneous_income
    
    def totalize_profit_loss(self):
        """
        損益通算し総所得を計算
        """
        self.calc_employment_income()
        self.calc_retirement_income()
        self.calc_temporary_income()
        self.calc_miscellaneous_income()
        
        total1 = self.interest_income + self.dividend_income + self.business_income + self.salary + self.miscellaneous_income
        #total1 += self.realestate_income
        total2 = self.miscellaneous_income
        #total2 += self.shortterm_capitalgai + self.longterm_capitalgain
        total3 = total1 + total2
        self.gross_income = total3 + self.retirement_income - self.employment_income_adjustment_deduction # + self.forestry_income 
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

    def calc_lifeinsurance_deduction(self, generallifeinsurance_premium=0, individualannuity_premium=0, medicallongtermcareinsurance_premium=0):
        """
        生命保険料控除
        """
        # 一般生命保険料控除
        self.calc_generallifeinsurance_deduction(generallifeinsurance_premium)
        # 個人年金保険料控除
        self.calc_individualannuity_deduction(individualannuity_premium)
        # 介護医療保険料控除
        self.calc_medicallongtermcareinsurance_deduction(medicallongtermcareinsurance_premium)
        
        self.lifeinsurance_deduction = self.generallifeinsurance_deduction + self.individualannuity_deduction + self.medicallongtermcareinsurance_deduction
        return self.lifeinsurance_deduction

    def calc_earthquakeinsurance_deduction(self, premium=0):
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
        # 医療費控除
        self.calc_medical_expenses_deduction()
        # 社会保険料控除
        self.calc_socialinsurance_deduction()
        # 小規模企業共済等掛金控除
        self.calc_smallmutual_deduction()
        # 生命保険料控除
        self.calc_lifeinsurance_deduction()
        # 地震保険料控除
        self.calc_earthquakeinsurance_deduction()
        # 寡婦・寡夫控除
        self.calc_widow_deduction()
        # 勤労学生控除控除
        self.calc_workingstudent_deduction()
        # 配偶者控除
        self.calc_spouse_deduction()
        # 扶養控除
        self.calc_dependents_deduction()
        # 障害者控除
        self.calc_disabledperson_deduction()
        # 寄付金控除
        self.calc_donation_deduction()
        self.income_deduction = min(self.basic_exemption + self.medical_expenses_deduction + self.socialinsurance_deduction + self.smallmutual_deduction \
            + self.lifeinsurance_deduction + self.earthquakeinsurance_deduction + self.donation_deduction \
            + self.disabledperson_deduction + self.widow_deduction + self.workingstudent_deduction \
            + self.spouse_deduction + self.dependents_deduction, self.gross_income)
        return self.income_deduction

    def calc_taxable_income(self):
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
        self.calc_dividend_deduction()
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

    def calc_tax(self,housing_loan_deduction=0):
        # 総所得
        self.totalize_profit_loss()
        # 所得控除
        self.calc_income_deduction()
        # 課税所得
        self.calc_taxable_income()
        # 算出税額
        self.calc_calculated_tax()
        # 税額控除
        self.calc_tax_credit(housing_loan_deduction)
        # 所得税
        self.calc_income_tax()
        return self.income_tax

    def calc_disposable_income(self):
        self.disposable_income = self.income - self.income_tax
        return self.disposable_income

        