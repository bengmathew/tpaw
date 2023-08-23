/**
 * @generated SignedSource<<0fc77fc82bcb3e821d2d9da4cfe234ec>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type UserPlanSyncInput = {
  add: ReadonlyArray<UserPlanSyncAddInput>;
  cutAfterId: string;
  lastSyncAt: number;
  planId: string;
  reverseHeadIndex: number;
  userId: string;
};
export type UserPlanSyncAddInput = {
  change: string;
  id: string;
  params: string;
};
export type UseServerSyncPlanMutation$variables = {
  input: UserPlanSyncInput;
};
export type UseServerSyncPlanMutation$data = {
  readonly userPlanSync: {
    readonly __typename: "ConcurrentChangeError";
    readonly _: number;
  } | {
    readonly __typename: "PlanAndUserResult";
    readonly plan: {
      readonly id: string;
      readonly lastSyncAt: number;
      readonly " $fragmentSpreads": FragmentRefs<"PlanWithoutParamsFragment">;
    };
  } | {
    // This will never be '%other', but we need some
    // value in case none of the concrete values match.
    readonly __typename: "%other";
  };
};
export type UseServerSyncPlanMutation = {
  response: UseServerSyncPlanMutation$data;
  variables: UseServerSyncPlanMutation$variables;
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
  "name": "__typename",
  "storageKey": null
},
v3 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "id",
  "storageKey": null
},
v4 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "lastSyncAt",
  "storageKey": null
},
v5 = {
  "kind": "InlineFragment",
  "selections": [
    {
      "alias": null,
      "args": null,
      "kind": "ScalarField",
      "name": "_",
      "storageKey": null
    }
  ],
  "type": "ConcurrentChangeError",
  "abstractKey": null
};
return {
  "fragment": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Fragment",
    "metadata": null,
    "name": "UseServerSyncPlanMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": null,
        "kind": "LinkedField",
        "name": "userPlanSync",
        "plural": false,
        "selections": [
          (v2/*: any*/),
          {
            "kind": "InlineFragment",
            "selections": [
              {
                "alias": null,
                "args": null,
                "concreteType": "PlanWithHistory",
                "kind": "LinkedField",
                "name": "plan",
                "plural": false,
                "selections": [
                  (v3/*: any*/),
                  (v4/*: any*/),
                  {
                    "args": null,
                    "kind": "FragmentSpread",
                    "name": "PlanWithoutParamsFragment"
                  }
                ],
                "storageKey": null
              }
            ],
            "type": "PlanAndUserResult",
            "abstractKey": null
          },
          (v5/*: any*/)
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
    "name": "UseServerSyncPlanMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": null,
        "kind": "LinkedField",
        "name": "userPlanSync",
        "plural": false,
        "selections": [
          (v2/*: any*/),
          {
            "kind": "InlineFragment",
            "selections": [
              {
                "alias": null,
                "args": null,
                "concreteType": "PlanWithHistory",
                "kind": "LinkedField",
                "name": "plan",
                "plural": false,
                "selections": [
                  (v3/*: any*/),
                  (v4/*: any*/),
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
                  }
                ],
                "storageKey": null
              }
            ],
            "type": "PlanAndUserResult",
            "abstractKey": null
          },
          (v5/*: any*/)
        ],
        "storageKey": null
      }
    ]
  },
  "params": {
    "cacheID": "98546ce4f87b07deaff95500d3937e76",
    "id": null,
    "metadata": {},
    "name": "UseServerSyncPlanMutation",
    "operationKind": "mutation",
    "text": "mutation UseServerSyncPlanMutation(\n  $input: UserPlanSyncInput!\n) {\n  userPlanSync(input: $input) {\n    __typename\n    ... on PlanAndUserResult {\n      plan {\n        id\n        lastSyncAt\n        ...PlanWithoutParamsFragment\n      }\n    }\n    ... on ConcurrentChangeError {\n      _\n    }\n  }\n}\n\nfragment PlanWithoutParamsFragment on PlanWithHistory {\n  id\n  isMain\n  label\n  slug\n  addedToServerAt\n  sortTime\n  lastSyncAt\n}\n"
  }
};
})();

(node as any).hash = "3d96ab521a4e34af42874e3326e5c648";

export default node;
