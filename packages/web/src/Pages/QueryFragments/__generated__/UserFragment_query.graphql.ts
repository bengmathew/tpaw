/**
 * @generated SignedSource<<406ee070e68dda8a71056bf73cab7be9>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { Fragment, ReaderFragment } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type UserFragment_query$data = {
  readonly user?: {
    readonly " $fragmentSpreads": FragmentRefs<"UserFragment_user">;
  };
  readonly " $fragmentType": "UserFragment_query";
};
export type UserFragment_query$key = {
  readonly " $data"?: UserFragment_query$data;
  readonly " $fragmentSpreads": FragmentRefs<"UserFragment_query">;
};

const node: ReaderFragment = {
  "argumentDefinitions": [
    {
      "kind": "RootArgument",
      "name": "includeUser"
    },
    {
      "kind": "RootArgument",
      "name": "userId"
    }
  ],
  "kind": "Fragment",
  "metadata": null,
  "name": "UserFragment_query",
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
            {
              "args": null,
              "kind": "FragmentSpread",
              "name": "UserFragment_user"
            }
          ],
          "storageKey": null
        }
      ]
    }
  ],
  "type": "Query",
  "abstractKey": null
};

(node as any).hash = "5a389293d64bf69b33cddc8bdeaed2ac";

export default node;
