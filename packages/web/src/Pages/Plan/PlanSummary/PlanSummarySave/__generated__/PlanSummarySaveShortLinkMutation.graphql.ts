/**
 * @generated SignedSource<<5b747e98bd22afdc350b8c43630f22da>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type CreateLinkBasedPlanInput = {
  params: string;
};
export type PlanSummarySaveShortLinkMutation$variables = {
  input: CreateLinkBasedPlanInput;
};
export type PlanSummarySaveShortLinkMutation$data = {
  readonly createLinkBasedPlan: {
    readonly id: string;
  };
};
export type PlanSummarySaveShortLinkMutation = {
  response: PlanSummarySaveShortLinkMutation$data;
  variables: PlanSummarySaveShortLinkMutation$variables;
};

const node: ConcreteRequest = (function(){
var v0 = [
  {
    "defaultValue": null,
    "kind": "LocalArgument",
    "name": "input"
  }
],
v1 = [
  {
    "alias": null,
    "args": [
      {
        "kind": "Variable",
        "name": "input",
        "variableName": "input"
      }
    ],
    "concreteType": "LinkBasedPlan",
    "kind": "LinkedField",
    "name": "createLinkBasedPlan",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "id",
        "storageKey": null
      }
    ],
    "storageKey": null
  }
];
return {
  "fragment": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Fragment",
    "metadata": null,
    "name": "PlanSummarySaveShortLinkMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "PlanSummarySaveShortLinkMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "6e575c642ce8885fc567b09e49bfbc00",
    "id": null,
    "metadata": {},
    "name": "PlanSummarySaveShortLinkMutation",
    "operationKind": "mutation",
    "text": "mutation PlanSummarySaveShortLinkMutation(\n  $input: CreateLinkBasedPlanInput!\n) {\n  createLinkBasedPlan(input: $input) {\n    id\n  }\n}\n"
  }
};
})();

(node as any).hash = "01a7b0a9302bc1b7da856fdb746cbad0";

export default node;
