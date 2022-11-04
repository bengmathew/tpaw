export declare function assert(condition: any): asserts condition;
export declare function assertFalse(): never;
export declare function fGet<T>(x: T | null | undefined): T;
export declare function noCase(x: never): never;
export declare function nundef<T>(x: T | undefined): T;
export declare const linearFnFomPoints: (x0: number, y0: number, x1: number, y1: number) => {
    (x: number): number;
    inverse(y: number): number;
};
export declare const linearFnFromPointAndSlope: (x: number, y: number, slope: number) => {
    (x: number): number;
    inverse(y: number): number;
};
export declare type LinearFn = ReturnType<typeof linearFnFromSlopeAndIntercept>;
export declare const linearFnFromSlopeAndIntercept: (slope: number, intercept: number) => {
    (x: number): number;
    inverse(y: number): number;
};
