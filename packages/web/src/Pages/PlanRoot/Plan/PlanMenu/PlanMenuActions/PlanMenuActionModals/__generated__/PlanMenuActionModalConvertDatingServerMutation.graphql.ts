/**
 * @generated SignedSource<<59674a97410f61d351054d002c0d3676>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type UserPlanResetInput = {
  lastSyncAt: number;
  planId: string;
  planParams: string;
  userId: string;
};
export type PlanMenuActionModalConvertDatingServerMutation$variables = {
  input: UserPlanResetInput;
};
export type PlanMenuActionModalConvertDatingServerMutation$data = {
  readonly userPlanReset: {
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
export type PlanMenuActionModalConvertDatingServerMutation = {
  response: PlanMenuActionModalConvertDatingServerMutation$data;
  variables: PlanMenuActionModalConvertDatingServerMutation$variables;
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
    "name": "PlanMenuActionModalConvertDatingServerMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": null,
        "kind": "LinkedField",
        "name": "userPlanReset",
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
    "name": "PlanMenuActionModalConvertDatingServerMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": null,
        "kind": "LinkedField",
        "name": "userPlanReset",
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
    "cacheID": "c93ee6a853e796149caa469edffdfe5b",
    "id": null,
    "metadata": {},
    "name": "PlanMenuActionModalConvertDatingServerMutation",
    "operationKind": "mutation",
    "text": "mutation PlanMenuActionModalConvertDatingServerMutation(\n  $input: UserPlanResetInput!\n) {\n  userPlanReset(input: $input) {\n    __typename\n    ... on PlanAndUserResult {\n      plan {\n        id\n        lastSyncAt\n        ...PlanWithoutParamsFragment\n      }\n    }\n    ... on ConcurrentChangeError {\n      _\n    }\n  }\n}\n\nfragment PlanWithoutParamsFragment on PlanWithHistory {\n  id\n  isMain\n  label\n  slug\n  addedToServerAt\n  sortTime\n  lastSyncAt\n}\n"
  }
};
})();

(node as any).hash = "c0f47b1ec1b0a08881237c34b6d0a61a";

export default node;
