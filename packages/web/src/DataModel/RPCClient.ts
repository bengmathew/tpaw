import { AppError } from '../Pages/App/AppError'
import { RPC } from '@tpaw/common'
import { FirebaseUser } from '../Pages/App/WithFirebaseUser'
import { Config } from '../Pages/Config'
import _ from 'lodash'

export type RPCClient = {
  [K in RPC.MethodName]: (
    args: RPC.Args<K>,
    firebaseUser: FirebaseUser,
    signal: AbortSignal | null,
  ) => Promise<RPC.Result<K>>
}

export const rpcClient: RPCClient = _.fromPairs(
  // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
  RPC.methodNames.map(
    (methodName) =>
      [
        methodName,
        async (
          args: RPC.Args<typeof methodName>,
          firebaseUser: FirebaseUser,
          signal: AbortSignal,
        ) => _impl(methodName, args, firebaseUser, signal),
      ] as const,
  ) as any,
) as RPCClient

const _impl = async <T extends RPC.MethodName>(
  method: T,
  args: RPC.Args<T>,
  firebaseUser: FirebaseUser,
  signal: AbortSignal | null,
): Promise<RPC.Result<T>> => {
  const authHeaders = _.compact([
    Config.client.debug.authHeader,
    `Bearer ${await firebaseUser.getIdToken(false)}`,
  ])

  let response
  try {
    response = await fetch(`${Config.client.urls.backend}/rpc/v1`, {
      signal,
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...(authHeaders ? { authorization: authHeaders.join(', ') } : {}),
      },
      body: JSON.stringify({ method, args }),
    })
  } catch (e) {
    throw new AppError('networkError')
  }
  if (!response.ok) {
    const code = response.headers.get('x-app-error-code')
    switch (code) {
      case 'downForMaintenance':
        throw new AppError('serverDownForMaintenance')
      case 'clientNeedsUpdate':
        throw new AppError('clientNeedsUpdate')
    }
    throw new AppError('networkError')
  }
  try {
    return (await response.json()) as RPC.Result<T>
  } catch (e) {
    throw new AppError('serverError')
  }
}
