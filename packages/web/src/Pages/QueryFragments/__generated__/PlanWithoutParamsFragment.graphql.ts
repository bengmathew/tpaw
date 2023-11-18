/**
 * @generated SignedSource<<832801b06600d08f9986b4051f087e0f>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { Fragment, ReaderFragment } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type PlanWithoutParamsFragment$data = {
  readonly addedToServerAt: number;
  readonly id: string;
  readonly isMain: boolean;
  readonly label: string | null | undefined;
  readonly lastSyncAt: number;
  readonly slug: string;
  readonly sortTime: number;
  readonly " $fragmentType": "PlanWithoutParamsFragment";
};
export type PlanWithoutParamsFragment$key = {
  readonly " $data"?: PlanWithoutParamsFragment$data;
  readonly " $fragmentSpreads": FragmentRefs<"PlanWithoutParamsFragment">;
};

const node: ReaderFragment = {
  "argumentDefinitions": [],
  "kind": "Fragment",
  "metadata": null,
  "name": "PlanWithoutParamsFragment",
  "selections": [
    {
      "alias": null,
      "args": null,
      "kind": "ScalarField",
      "name": "id",
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
    }
  ],
  "type": "PlanWithHistory",
  "abstractKey": null
};

(node as any).hash = "5e521444f31ee1ab5b4782e6625f9bfb";

export default node;
