/**
 * @generated SignedSource<<aad378d00a52006ca8b06f167c91e41c>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest } from 'relay-runtime';
export type CreateLinkBasedPlanInput = {
  params: string;
};
export type PlanMenuActionCopyToLinkShortLinkMutation$variables = {
  input: CreateLinkBasedPlanInput;
};
export type PlanMenuActionCopyToLinkShortLinkMutation$data = {
  readonly createLinkBasedPlan: {
    readonly id: string;
  };
};
export type PlanMenuActionCopyToLinkShortLinkMutation = {
  response: PlanMenuActionCopyToLinkShortLinkMutation$data;
  variables: PlanMenuActionCopyToLinkShortLinkMutation$variables;
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
    "name": "PlanMenuActionCopyToLinkShortLinkMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "PlanMenuActionCopyToLinkShortLinkMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "503c6cdaaa2672a5da705129535e3f4a",
    "id": null,
    "metadata": {},
    "name": "PlanMenuActionCopyToLinkShortLinkMutation",
    "operationKind": "mutation",
    "text": "mutation PlanMenuActionCopyToLinkShortLinkMutation(\n  $input: CreateLinkBasedPlanInput!\n) {\n  createLinkBasedPlan(input: $input) {\n    id\n  }\n}\n"
  }
};
})();

(node as any).hash = "3344edcc946491f9a51ad84d3e7dce4c";

export default node;
