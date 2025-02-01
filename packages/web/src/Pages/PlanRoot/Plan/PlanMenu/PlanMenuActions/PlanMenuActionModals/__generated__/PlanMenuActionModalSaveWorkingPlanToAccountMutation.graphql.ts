/**
 * @generated SignedSource<<44020da371971bd3ef84b64bc672321a>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type UserPlanCreateInput = {
  label: string;
  plan: UserPlanCreatePlanInput;
  userId: string;
};
export type UserPlanCreatePlanInput = {
  planParamsHistory: ReadonlyArray<UserPlanCreatePlanParamsHistryInput>;
  reverseHeadIndex: number;
};
export type UserPlanCreatePlanParamsHistryInput = {
  change: string;
  id: string;
  params: string;
};
export type PlanMenuActionModalSaveWorkingPlanToAccountMutation$variables = {
  input: UserPlanCreateInput;
};
export type PlanMenuActionModalSaveWorkingPlanToAccountMutation$data = {
  readonly userPlanCreate: {
    readonly plan: {
      readonly slug: string;
      readonly " $fragmentSpreads": FragmentRefs<"PlanWithoutParamsFragment">;
    };
    readonly user: {
      readonly " $fragmentSpreads": FragmentRefs<"WithUser_user">;
    };
  };
};
export type PlanMenuActionModalSaveWorkingPlanToAccountMutation = {
  response: PlanMenuActionModalSaveWorkingPlanToAccountMutation$data;
  variables: PlanMenuActionModalSaveWorkingPlanToAccountMutation$variables;
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
  "name": "slug",
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
  "name": "isMain",
  "storageKey": null
},
v5 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "label",
  "storageKey": null
},
v6 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "addedToServerAt",
  "storageKey": null
},
v7 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "sortTime",
  "storageKey": null
},
v8 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "lastSyncAt",
  "storageKey": null
};
return {
  "fragment": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Fragment",
    "metadata": null,
    "name": "PlanMenuActionModalSaveWorkingPlanToAccountMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": "PlanAndUserResult",
        "kind": "LinkedField",
        "name": "userPlanCreate",
        "plural": false,
        "selections": [
          {
            "alias": null,
            "args": null,
            "concreteType": "PlanWithHistory",
            "kind": "LinkedField",
            "name": "plan",
            "plural": false,
            "selections": [
              (v2/*: any*/),
              {
                "args": null,
                "kind": "FragmentSpread",
                "name": "PlanWithoutParamsFragment"
              }
            ],
            "storageKey": null
          },
          {
            "alias": null,
            "args": null,
            "concreteType": "User",
            "kind": "LinkedField",
            "name": "user",
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
    "name": "PlanMenuActionModalSaveWorkingPlanToAccountMutation",
    "selections": [
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": "PlanAndUserResult",
        "kind": "LinkedField",
        "name": "userPlanCreate",
        "plural": false,
        "selections": [
          {
            "alias": null,
            "args": null,
            "concreteType": "PlanWithHistory",
            "kind": "LinkedField",
            "name": "plan",
            "plural": false,
            "selections": [
              (v2/*: any*/),
              (v3/*: any*/),
              (v4/*: any*/),
              (v5/*: any*/),
              (v6/*: any*/),
              (v7/*: any*/),
              (v8/*: any*/)
            ],
            "storageKey": null
          },
          {
            "alias": null,
            "args": null,
            "concreteType": "User",
            "kind": "LinkedField",
            "name": "user",
            "plural": false,
            "selections": [
              (v3/*: any*/),
              {
                "alias": null,
                "args": null,
                "concreteType": "PlanWithHistory",
                "kind": "LinkedField",
                "name": "plans",
                "plural": true,
                "selections": [
                  (v3/*: any*/),
                  (v5/*: any*/),
                  (v2/*: any*/),
                  (v6/*: any*/),
                  (v7/*: any*/),
                  (v8/*: any*/),
                  (v4/*: any*/),
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
        ],
        "storageKey": null
      }
    ]
  },
  "params": {
    "cacheID": "7cb7f94b234e59916cdf49778ba0a3ce",
    "id": null,
    "metadata": {},
    "name": "PlanMenuActionModalSaveWorkingPlanToAccountMutation",
    "operationKind": "mutation",
    "text": "mutation PlanMenuActionModalSaveWorkingPlanToAccountMutation(\n  $input: UserPlanCreateInput!\n) {\n  userPlanCreate(input: $input) {\n    plan {\n      slug\n      ...PlanWithoutParamsFragment\n      id\n    }\n    user {\n      ...WithUser_user\n      id\n    }\n  }\n}\n\nfragment PlanWithoutParamsFragment on PlanWithHistory {\n  id\n  isMain\n  label\n  slug\n  addedToServerAt\n  sortTime\n  lastSyncAt\n}\n\nfragment WithUser_user on User {\n  id\n  plans {\n    id\n    label\n    slug\n    addedToServerAt\n    sortTime\n    lastSyncAt\n    isMain\n    isDated\n  }\n  nonPlanParamsLastUpdatedAt\n  nonPlanParams\n}\n"
  }
};
})();

(node as any).hash = "f03eb6ff0eff3d9411117e86b1d3e484";

export default node;
