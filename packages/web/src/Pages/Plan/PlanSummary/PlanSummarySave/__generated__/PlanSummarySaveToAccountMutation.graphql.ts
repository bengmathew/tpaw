/**
 * @generated SignedSource<<1e638e66d70315ae8e7c1e5f5acc7abe>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type SetUserPlanInput = {
  params: string;
  userId: string;
};
export type PlanSummarySaveToAccountMutation$variables = {
  input: SetUserPlanInput;
};
export type PlanSummarySaveToAccountMutation$data = {
  readonly setUserPlan: {
    readonly " $fragmentSpreads": FragmentRefs<"UserFragment_user">;
  };
};
export type PlanSummarySaveToAccountMutation = {
  response: PlanSummarySaveToAccountMutation$data;
  variables: PlanSummarySaveToAccountMutation$variables;
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
    "kind": "Variable",
    "name": "input",
    "variableName": "input"
  }
],
v2 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "id",
  "storageKey": null
};
return {
  "fragment": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Fragment",
    "metadata": null,
    "name": "PlanSummarySaveToAccountMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": "User",
        "kind": "LinkedField",
        "name": "setUserPlan",
        "plural": false,
        "selections": [
          {
            "args": null,
            "kind": "FragmentSpread",
            "name": "UserFragment_user"
          }
        ],
        "storageKey": null
      }
    ],
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "PlanSummarySaveToAccountMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": "User",
        "kind": "LinkedField",
        "name": "setUserPlan",
        "plural": false,
        "selections": [
          (v2/*: any*/),
          {
            "alias": null,
            "args": null,
            "concreteType": "Plan",
            "kind": "LinkedField",
            "name": "plan",
            "plural": false,
            "selections": [
              (v2/*: any*/),
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "createdAt",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "modifiedAt",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "params",
                "storageKey": null
              }
            ],
            "storageKey": null
          }
        ],
        "storageKey": null
      }
    ]
  },
  "params": {
    "cacheID": "68d95acf63cd8f6045c60de0a91918ef",
    "id": null,
    "metadata": {},
    "name": "PlanSummarySaveToAccountMutation",
    "operationKind": "mutation",
    "text": "mutation PlanSummarySaveToAccountMutation(\n  $input: SetUserPlanInput!\n) {\n  setUserPlan(input: $input) {\n    ...UserFragment_user\n    id\n  }\n}\n\nfragment UserFragment_user on User {\n  id\n  plan {\n    id\n    createdAt\n    modifiedAt\n    params\n  }\n}\n"
  }
};
})();

(node as any).hash = "4abe6c1eceafe74d373d80a40fb47850";

export default node;
