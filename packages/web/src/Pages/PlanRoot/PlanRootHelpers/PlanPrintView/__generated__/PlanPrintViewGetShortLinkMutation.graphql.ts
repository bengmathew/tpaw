/**
 * @generated SignedSource<<7460272307479dc1a12bb250ac685e8d>>
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
export type PlanPrintViewGetShortLinkMutation$variables = {
  input: CreateLinkBasedPlanInput;
};
export type PlanPrintViewGetShortLinkMutation$data = {
  readonly createLinkBasedPlan: {
    readonly id: string;
  };
};
export type PlanPrintViewGetShortLinkMutation = {
  response: PlanPrintViewGetShortLinkMutation$data;
  variables: PlanPrintViewGetShortLinkMutation$variables;
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
    "name": "PlanPrintViewGetShortLinkMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "PlanPrintViewGetShortLinkMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "f6ced4e475e02ebf143ee1054ad86ccd",
    "id": null,
    "metadata": {},
    "name": "PlanPrintViewGetShortLinkMutation",
    "operationKind": "mutation",
    "text": "mutation PlanPrintViewGetShortLinkMutation(\n  $input: CreateLinkBasedPlanInput!\n) {\n  createLinkBasedPlan(input: $input) {\n    id\n  }\n}\n"
  }
};
})();

(node as any).hash = "ddc56cf045fa6ccd5e913706d0c43970";

export default node;
