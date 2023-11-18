/**
 * @generated SignedSource<<463618cb39b08ea8c8067fb0ac3e6220>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Query } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type PlanRootLinkQuery$variables = {
  includeLink: boolean;
  includeUser: boolean;
  linkId: string;
  userId: string;
};
export type PlanRootLinkQuery$data = {
  readonly linkBasedPlan?: {
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
  "name": "includeLink"
},
v1 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "includeUser"
},
v2 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "linkId"
},
v3 = {
  "defaultValue": null,
  "kind": "LocalArgument",
  "name": "userId"
},
v4 = [
  {
    "kind": "Variable",
    "name": "linkId",
    "variableName": "linkId"
  }
],
v5 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "params",
  "storageKey": null
},
v6 = {
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
      (v2/*: any*/),
      (v3/*: any*/)
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
        "condition": "includeLink",
        "kind": "Condition",
        "passingValue": true,
        "selections": [
          {
            "alias": null,
            "args": (v4/*: any*/),
            "concreteType": "LinkBasedPlan",
            "kind": "LinkedField",
            "name": "linkBasedPlan",
            "plural": false,
            "selections": [
              (v5/*: any*/)
            ],
            "storageKey": null
          }
        ]
      }
    ],
    "type": "Query",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": [
      (v3/*: any*/),
      (v1/*: any*/),
      (v2/*: any*/),
      (v0/*: any*/)
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
              (v6/*: any*/),
              {
                "alias": null,
                "args": null,
                "concreteType": "PlanWithHistory",
                "kind": "LinkedField",
                "name": "plans",
                "plural": true,
                "selections": [
                  (v6/*: any*/),
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
        "condition": "includeLink",
        "kind": "Condition",
        "passingValue": true,
        "selections": [
          {
            "alias": null,
            "args": (v4/*: any*/),
            "concreteType": "LinkBasedPlan",
            "kind": "LinkedField",
            "name": "linkBasedPlan",
            "plural": false,
            "selections": [
              (v5/*: any*/),
              (v6/*: any*/)
            ],
            "storageKey": null
          }
        ]
      }
    ]
  },
  "params": {
    "cacheID": "e34e7f5bf2eb04699a73b3fe7601d477",
    "id": null,
    "metadata": {},
    "name": "PlanRootLinkQuery",
    "operationKind": "query",
    "text": "query PlanRootLinkQuery(\n  $userId: ID!\n  $includeUser: Boolean!\n  $linkId: ID!\n  $includeLink: Boolean!\n) {\n  ...WithUser_query\n  linkBasedPlan(linkId: $linkId) @include(if: $includeLink) {\n    params\n    id\n  }\n}\n\nfragment WithUser_query on Query {\n  user(userId: $userId) @include(if: $includeUser) {\n    ...WithUser_user\n    id\n  }\n}\n\nfragment WithUser_user on User {\n  id\n  plans {\n    id\n    label\n    slug\n    addedToServerAt\n    sortTime\n    lastSyncAt\n    isMain\n  }\n  nonPlanParamsLastUpdatedAt\n  nonPlanParams\n}\n"
  }
};
})();

(node as any).hash = "5b62182559d0f90d4af1b65805e60f66";

export default node;
