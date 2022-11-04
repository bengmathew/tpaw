"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.API = void 0;
const json_guard_1 = require("json-guard");
const PlanParams_1 = require("./PlanParams/PlanParams");
var API;
(function (API) {
    const trimmed = (x) => x.trim().length === x.length
        ? (0, json_guard_1.success)(x)
        : (0, json_guard_1.failure)('String is not trimmed.');
    const nonEmpty = (x) => x.length > 0 ? (0, json_guard_1.success)(x) : (0, json_guard_1.failure)('Empty string.');
    const email = (0, json_guard_1.chain)(json_guard_1.string, trimmed, (x) => {
        const EMAIL_REGEX = /^[^@]+@([^@]+\.[^@]+)$/;
        const DNS_REGEX = /^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$/;
        const emailMatch = EMAIL_REGEX.exec(x);
        if (emailMatch === null || !emailMatch[1])
            return (0, json_guard_1.failure)('Email is invalid.');
        if (!DNS_REGEX.test(emailMatch[1]))
            return (0, json_guard_1.failure)('DNS part of email is invalid');
        return (0, json_guard_1.success)(x);
    });
    const userId = (0, json_guard_1.chain)(json_guard_1.string, (0, json_guard_1.bounded)(100));
    let SendSignInEmail;
    (function (SendSignInEmail) {
        SendSignInEmail.guards = { email, dest: json_guard_1.string };
        SendSignInEmail.check = (0, json_guard_1.object)(SendSignInEmail.guards);
    })(SendSignInEmail = API.SendSignInEmail || (API.SendSignInEmail = {}));
    let SetUserPlan;
    (function (SetUserPlan) {
        SetUserPlan.check = (0, json_guard_1.object)({
            userId,
            params: (0, json_guard_1.chain)(json_guard_1.string, json_guard_1.json, PlanParams_1.planParamsGuard),
        });
    })(SetUserPlan = API.SetUserPlan || (API.SetUserPlan = {}));
    let CreateLinkBasedPlan;
    (function (CreateLinkBasedPlan) {
        CreateLinkBasedPlan.check = (0, json_guard_1.object)({
            params: (0, json_guard_1.chain)(json_guard_1.string, json_guard_1.json, PlanParams_1.planParamsGuard),
        });
    })(CreateLinkBasedPlan = API.CreateLinkBasedPlan || (API.CreateLinkBasedPlan = {}));
})(API = exports.API || (exports.API = {}));
//# sourceMappingURL=API.js.map