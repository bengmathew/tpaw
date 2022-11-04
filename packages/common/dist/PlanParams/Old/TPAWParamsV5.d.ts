import { Validator } from './Validator';
export declare namespace TPAWParamsV5 {
    const MAX_LABEL_LENGTH = 150;
    const MAX_AGE = 120;
    type Person = {
        ages: {
            type: 'retired';
            current: number;
            max: number;
        } | {
            type: 'notRetired';
            current: number;
            retirement: number;
            max: number;
        };
        displayName: string | null;
    };
    type People = {
        withPartner: false;
        person1: Person;
    } | {
        withPartner: true;
        person2: Person;
        person1: Person;
        withdrawalStart: 'person1' | 'person2';
        xAxis: 'person1' | 'person2';
    };
    type Year = {
        type: 'now';
    } | {
        type: 'namedAge';
        person: 'person1' | 'person2';
        age: 'lastWorkingYear' | 'retirement' | 'max';
    } | {
        type: 'numericAge';
        person: 'person1' | 'person2';
        age: number;
    };
    type YearRange = {
        type: 'startAndEnd';
        start: Year;
        end: Year;
    } | {
        type: 'startAndNumYears';
        start: Year;
        numYears: number;
    } | {
        type: 'endAndNumYears';
        end: Year;
        numYears: number;
    };
    type ValueForYearRange = {
        label: string | null;
        yearRange: YearRange;
        value: number;
        nominal: boolean;
        id: number;
    };
    type LabeledAmount = {
        label: string | null;
        value: number;
        nominal: boolean;
    };
    type Params = {
        v: 5;
        people: People;
        returns: {
            expected: {
                stocks: number;
                bonds: number;
            };
            historical: {
                adjust: {
                    type: 'by';
                    stocks: number;
                    bonds: number;
                } | {
                    type: 'to';
                    stocks: number;
                    bonds: number;
                };
            };
        };
        inflation: number;
        targetAllocation: {
            regularPortfolio: {
                stocks: number;
            };
            legacyPortfolio: {
                stocks: number;
            };
        };
        spendingCeiling: number | null;
        spendingFloor: number | null;
        scheduledWithdrawalGrowthRate: number;
        savingsAtStartOfStartYear: number;
        savings: ValueForYearRange[];
        retirementIncome: ValueForYearRange[];
        withdrawals: {
            fundedByBonds: ValueForYearRange[];
            fundedByRiskPortfolio: ValueForYearRange[];
        };
        legacy: {
            total: number;
            external: LabeledAmount[];
        };
    };
    type ParamsWithoutHistorical = Omit<Params, 'returns'> & {
        returns: Omit<Params['returns'], 'historical'>;
    };
    const validator: Validator<TPAWParamsV5.ParamsWithoutHistorical>;
}
