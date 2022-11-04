import { JSONGuard } from 'json-guard';
export declare namespace API {
    namespace SendSignInEmail {
        const guards: {
            email: JSONGuard<string, unknown>;
            dest: JSONGuard<string, unknown>;
        };
        const check: (x: unknown) => import("json-guard").JSONGuardResult<import("json-guard/dist/utils").GuardToSuccessType<{
            email: JSONGuard<string, unknown>;
            dest: JSONGuard<string, unknown>;
        }>>;
    }
    namespace SetUserPlan {
        const check: (x: unknown) => import("json-guard").JSONGuardResult<import("json-guard/dist/utils").GuardToSuccessType<{
            userId: JSONGuard<string, unknown>;
            params: JSONGuard<import("./PlanParams/PlanParams14").PlanParams14.Params, unknown>;
        }>>;
    }
    namespace CreateLinkBasedPlan {
        const check: (x: unknown) => import("json-guard").JSONGuardResult<import("json-guard/dist/utils").GuardToSuccessType<{
            params: JSONGuard<import("./PlanParams/PlanParams14").PlanParams14.Params, unknown>;
        }>>;
    }
}
