/**
 * @generated SignedSource<<1b9f63e86b27d3a59e5d8579d4f64768>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Query } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type PlanRootLinkQuery$variables = {
  includeUser: boolean;
  linkId: string;
  userId: string;
};
export type PlanRootLinkQuery$data = {
  readonly linkBasedPlan: {
    readonly params: string;
  } | null | undefined;
  readonly " $fragmentSpreads": FragmentRefs<"WithUser_query">;
};
export type PlanRootLinkQuery = {
  response: PlanRootLinkQuery$data;
  variables: PlanRootLinkQuery$variables;
};

const node: ConcreteRequest = (function(){
var v0 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "includeUser"
},
v1 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "linkId"
},
v2 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "userId"
},
v3 = [
  {
    "kind": "Variable",
    "name": "linkId",
    "variableName": "linkId"
  }
],
v4 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "params",
  "storageKey": null
},
v5 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "id",
  "storageKey": null
};
return {
  "fragment": {
    "argumentDefinitions": [
      (v0/*: any*/),
      (v1/*: any*/),
      (v2/*: any*/)
    ],
    "kind": "Fragment",
    "metadata": null,
    "name": "PlanRootLinkQuery",
    "selections": [
      {
        "args": null,
        "kind": "FragmentSpread",
        "name": "WithUser_query"
      },
      {
        "alias": null,
        "args": (v3/*: any*/),
        "concreteType": "LinkBasedPlan",
        "kind": "LinkedField",
        "name": "linkBasedPlan",
        "plural": false,
        "selections": [
          (v4/*: any*/)
        ],
        "storageKey": null
      }
    ],
    "type": "Query",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": [
      (v2/*: any*/),
      (v0/*: any*/),
      (v1/*: any*/)
    ],
    "kind": "Operation",
    "name": "PlanRootLinkQuery",
    "selections": [
      {
        "condition": "includeUser",
        "kind": "Condition",
        "passingValue": true,
        "selections": [
          {
            "alias": null,
            "args": [
              {
                "kind": "Variable",
                "name": "userId",
                "variableName": "userId"
              }
            ],
            "concreteType": "User",
            "kind": "LinkedField",
            "name": "user",
            "plural": false,
            "selections": [
              (v5/*: any*/),
              {
                "alias": null,
                "args": null,
                "concreteType": "PlanWithHistory",
                "kind": "LinkedField",
                "name": "plans",
                "plural": true,
                "selections": [
                  (v5/*: any*/),
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
      {
        "alias": null,
        "args": (v3/*: any*/),
        "concreteType": "LinkBasedPlan",
        "kind": "LinkedField",
        "name": "linkBasedPlan",
        "plural": false,
        "selections": [
          (v4/*: any*/),
          (v5/*: any*/)
        ],
        "storageKey": null
      }
    ]
  },
  "params": {
    "cacheID": "911193029660ba3ea9151eb78de12f66",
    "id": null,
    "metadata": {},
    "name": "PlanRootLinkQuery",
    "operationKind": "query",
    "text": "query PlanRootLinkQuery(\n  $userId: ID!\n  $includeUser: Boolean!\n  $linkId: ID!\n) {\n  ...WithUser_query\n  linkBasedPlan(linkId: $linkId) {\n    params\n    id\n  }\n}\n\nfragment WithUser_query on Query {\n  user(userId: $userId) @include(if: $includeUser) {\n    ...WithUser_user\n    id\n  }\n}\n\nfragment WithUser_user on User {\n  id\n  plans {\n    id\n    label\n    slug\n    addedToServerAt\n    sortTime\n    lastSyncAt\n    isMain\n    isDated\n  }\n  nonPlanParamsLastUpdatedAt\n  nonPlanParams\n}\n"
  }
};
})();

(node as any).hash = "76beb3b8c9c16b130379d0d24c55296c";

export default node;
