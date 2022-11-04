/**
 * @generated SignedSource<<d25171bd3cf93333b84139117a1c7c64>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type SendSignInEmailInput = {
  dest: string;
  email: string;
};
export type RequireUserSendEmailMutation$variables = {
  input: SendSignInEmailInput;
};
export type RequireUserSendEmailMutation$data = {
  readonly sendSignInEmail: {
    readonly __typename: "Success";
    readonly _: number;
  };
};
export type RequireUserSendEmailMutation = {
  response: RequireUserSendEmailMutation$data;
  variables: RequireUserSendEmailMutation$variables;
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
    "alias": null,
    "args": [
      {
        "kind": "Variable",
        "name": "input",
        "variableName": "input"
      }
    ],
    "concreteType": "Success",
    "kind": "LinkedField",
    "name": "sendSignInEmail",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "__typename",
        "storageKey": null
      },
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "_",
        "storageKey": null
      }
    ],
    "storageKey": null
  }
];
return {
  "fragment": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Fragment",
    "metadata": null,
    "name": "RequireUserSendEmailMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "RequireUserSendEmailMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "d7a19ea887244d759de1f412dd6b9bc2",
    "id": null,
    "metadata": {},
    "name": "RequireUserSendEmailMutation",
    "operationKind": "mutation",
    "text": "mutation RequireUserSendEmailMutation(\n  $input: SendSignInEmailInput!\n) {\n  sendSignInEmail(input: $input) {\n    __typename\n    _\n  }\n}\n"
  }
};
})();

(node as any).hash = "91d02cd1c4e3ead004befc21dd035406";

export default node;
