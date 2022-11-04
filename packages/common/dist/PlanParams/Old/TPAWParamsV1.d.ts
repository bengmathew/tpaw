declare type YearRangeFixedEdge = 'start' | 'lastWorkingYear' | 'retirement' | 'end';
declare type YearRangeEdge = number | YearRangeFixedEdge;
declare type YearRange = {
    start: YearRangeEdge;
    end: YearRangeEdge;
};
declare type ValueForYearRange = {
    label: string | null;
    yearRange: YearRange;
    value: number;
    nominal: boolean;
};
export declare type TPAWParamsV1 = {
    age: {
        start: number;
        retirement: number;
        end: number;
    };
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
    scheduledWithdrawalGrowthRate: number;
    savingsAtStartOfStartYear: number;
    savings: ValueForYearRange[];
    withdrawals: {
        fundedByBonds: ValueForYearRange[];
        fundedByRiskPortfolio: ValueForYearRange[];
    };
    legacy: {
        total: number;
        external: {
            label: string | null;
            value: number;
            nominal: boolean;
        }[];
    };
};
export declare type TPAWParamsV1WithoutHistorical = Omit<TPAWParamsV1, 'returns'> & {
    returns: Omit<TPAWParamsV1['returns'], 'historical'>;
};
export {};
