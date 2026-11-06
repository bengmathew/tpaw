import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

// This is for issue:
// https://github.com/getsentry/sentry-javascript/issues/8341. Using workaround
// in comment:
// https://github.com/getsentry/sentry-javascript/issues/8341#issuecomment-2449890027.
// Summary: Sentry tunnel does not work well with the authorization header used
// for development environment. This strips the header for the tunnel.
export function middleware(request: NextRequest) {
  const headers = new Headers(request.headers)

  headers.delete('authorization')

  return NextResponse.next({
    request: {
      headers: headers,
    },
  })
}

export const config = {
  matcher: [
    {
      // Must match tunnelRoute in next.config.js.
      source: '/__ingress',
      has: [{ type: 'header', key: 'authorization' }],
    },
  ],
}
