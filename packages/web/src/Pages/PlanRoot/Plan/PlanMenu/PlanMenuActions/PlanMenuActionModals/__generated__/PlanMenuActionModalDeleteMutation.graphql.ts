/**
 * @generated SignedSource<<690a1fe20eb9b27efd98d803cc239fbc>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type UserPlanDeleteInput = {
  planId: string;
  userId: string;
};
export type PlanMenuActionModalDeleteMutation$variables = {
  input: UserPlanDeleteInput;
};
export type PlanMenuActionModalDeleteMutation$data = {
  readonly userPlanDelete: {
    readonly " $fragmentSpreads": FragmentRefs<"WithUser_user">;
  };
};
export type PlanMenuActionModalDeleteMutation = {
  response: PlanMenuActionModalDeleteMutation$data;
  variables: PlanMenuActionModalDeleteMutation$variables;
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
    "name": "PlanMenuActionModalDeleteMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": "User",
        "kind": "LinkedField",
        "name": "userPlanDelete",
        "plural": false,
        "selections": [
          {
            "args": null,
            "kind": "FragmentSpread",
            "name": "WithUser_user"
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
    "name": "PlanMenuActionModalDeleteMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": "User",
        "kind": "LinkedField",
        "name": "userPlanDelete",
        "plural": false,
        "selections": [
          (v2/*: any*/),
          {
            "alias": null,
            "args": null,
            "concreteType": "PlanWithHistory",
            "kind": "LinkedField",
            "name": "plans",
            "plural": true,
            "selections": [
              (v2/*: any*/),
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "label",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "slug",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "addedToServerAt",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "sortTime",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "lastSyncAt",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "isMain",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "isDated",
                "storageKey": null
              }
            ],
            "storageKey": null
          },
          {
            "alias": null,
            "args": null,
            "kind": "ScalarField",
            "name": "nonPlanParamsLastUpdatedAt",
            "storageKey": null
          },
          {
            "alias": null,
            "args": null,
            "kind": "ScalarField",
            "name": "nonPlanParams",
            "storageKey": null
          }
        ],
        "storageKey": null
      }
    ]
  },
  "params": {
    "cacheID": "170920ab0bba26c77cb94bb2d22b4300",
    "id": null,
    "metadata": {},
    "name": "PlanMenuActionModalDeleteMutation",
    "operationKind": "mutation",
    "text": "mutation PlanMenuActionModalDeleteMutation(\n  $input: UserPlanDeleteInput!\n) {\n  userPlanDelete(input: $input) {\n    ...WithUser_user\n    id\n  }\n}\n\nfragment WithUser_user on User {\n  id\n  plans {\n    id\n    label\n    slug\n    addedToServerAt\n    sortTime\n    lastSyncAt\n    isMain\n    isDated\n  }\n  nonPlanParamsLastUpdatedAt\n  nonPlanParams\n}\n"
  }
};
})();

(node as any).hash = "41e16ae2c981549fd731c90bb1064363";

export default node;
