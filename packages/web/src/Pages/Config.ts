import {fGet} from '../Utils/Utils'

export class Config {
  static get server() {
    return {}
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
