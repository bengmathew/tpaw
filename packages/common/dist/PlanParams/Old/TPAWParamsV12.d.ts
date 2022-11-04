import { Validator } from './Validator';
import { TPAWParamsV11 } from './TPAWParamsV11';
export declare namespace TPAWParamsV12 {
    const MAX_LABEL_LENGTH = 150;
    const MAX_AGE = 120;
    const MAX_NUM_YEARS_IN_GLIDE_PATH = 1000;
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
    type GlidePath = {
        start: {
            stocks: number;
        };
        intermediate: {
            year: Year;
            stocks: number;
        }[];
        end: {
            stocks: number;
        };
    };
    type Params = {
        v: 12;
        strategy: 'TPAW' | 'SPAW' | 'SWR';
        people: People;
        returns: {
            expected: {
                type: 'suggested';
            } | {
                type: 'oneOverCAPE';
            } | {
                type: 'regressionPrediction';
            } | {
                type: 'historical';
            } | {
                type: 'manual';
                stocks: number;
                bonds: number;
            };
            historical: {
                type: 'default';
                adjust: {
                    type: 'by';
                    stocks: number;
                    bonds: number;
                } | {
                    type: 'to';
                    stocks: number;
                    bonds: number;
                } | {
                    type: 'toExpected';
                } | {
                    type: 'none';
                };
            } | {
                type: 'fixed';
                stocks: number;
                bonds: number;
            };
        };
        inflation: {
            type: 'suggested';
        } | {
            type: 'manual';
            value: number;
        };
        targetAllocation: {
            regularPortfolio: {
                forTPAW: GlidePath;
                forSPAWAndSWR: GlidePath;
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
            lmp: number;
            essential: ValueForYearRange[];
            discretionary: ValueForYearRange[];
        };
        swrWithdrawal: {
            type: 'asPercent';
            percent: number;
        } | {
            type: 'asAmount';
            amount: number;
        } | {
            type: 'default';
        };
        legacy: {
            total: number;
            external: LabeledAmount[];
        };
        sampling: 'monteCarlo' | 'historical';
        display: {
            alwaysShowAllYears: boolean;
        };
    };
    const fromV11: (params: TPAWParamsV11.Params) => Params;
    const validator: Validator<Params>;
}
