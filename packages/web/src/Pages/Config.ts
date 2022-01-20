import { fGet } from '../Utils/Utils'

export class Config {
  static get server() {
    return {
      contentful: {
        space: `${fGet(process.env.CONTENTFUL_SPACE_ID)}`,
        accessToken: `${fGet(process.env.CONTENTFUL_ACCESS_TOKEN)}`,
      },
    }
  }

  static get client() {
    return {
      urls: {
        app: () => `${fGet(process.env.NEXT_PUBLIC_URL_WEBSITE)}}`,
      },
      production: process.env.NODE_ENV === 'production',
      debug: {},
    }
  }
}
