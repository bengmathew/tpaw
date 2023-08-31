import * as process from 'process'
import { fGet } from '../Utils/Utils'

export class Config {
  static get server() {
    return {
      contentful: {
        space: fGet(process.env.CONTENTFUL_SPACE_ID),
        accessToken: fGet(process.env.CONTENTFUL_ACCESS_TOKEN),
      },
    }
  }

  static get client() {
    return {
      urls: {
        app: (path = '') =>
          `${fGet(process.env.NEXT_PUBLIC_URL_WEBSITE)}${path}`,
        backend: fGet(process.env.NEXT_PUBLIC_URL_BACKEND),
      },
      production: process.env.NODE_ENV === 'production',
      google: {
        cloud: {
          projectId: fGet(process.env.NEXT_PUBLIC_GOOGLE_CLOUD_PROJECT_ID),
        },
        firebase: {
          apiKey: fGet(process.env.NEXT_PUBLIC_GOOGLE_FIREBASE_API_KEY),
          appId: fGet(process.env.NEXT_PUBLIC_GOOGLE_FIREBASE_APP_ID),
          authDomain: fGet(process.env.NEXT_PUBLIC_GOOGLE_FIREBASE_AUTH_DOMAIN),
          storageBucket: fGet(
            process.env.NEXT_PUBLIC_GOOGLE_FIREBASE_STORAGE_BUCKET,
          ),
          messagingSenderId: fGet(
            process.env.NEXT_PUBLIC_GOOGLE_FIREBASE_MESSAGING_SENDER_ID,
          ),
        },
      },
      debug: {
        authHeader: process.env.NEXT_PUBLIC_DEBUG_AUTH_HEADER,
      },
    }
  }
}
