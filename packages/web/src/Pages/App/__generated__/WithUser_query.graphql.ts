/**
 * @generated SignedSource<<be2533f45159e18ed9c107850dbc1d43>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ReaderFragment } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type WithUser_query$data = {
  readonly user?: {
    readonly " $fragmentSpreads": FragmentRefs<"WithUser_user">;
  };
  readonly " $fragmentType": "WithUser_query";
};
export type WithUser_query$key = {
  readonly " $data"?: WithUser_query$data;
  readonly " $fragmentSpreads": FragmentRefs<"WithUser_query">;
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
  "name": "WithUser_query",
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
              "name": "WithUser_user"
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

(node as any).hash = "5bf0e1a2f2a70f6f5dc683e90af66770";

export default node;
