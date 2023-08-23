import { Storage } from '@google-cloud/storage'
import { PrismaClient } from '@prisma/client'
import firebase from 'firebase-admin'
import { applicationDefault } from 'firebase-admin/app'
import { getAuth } from 'firebase-admin/auth'
import postmark from 'postmark'
import { Config } from './Config.js'

export class Clients {
  private static _prisma: PrismaClient | null = null
  static get prisma() {
    if (!this._prisma) this._prisma = new PrismaClient()
    return this._prisma
  }

  private static _postmark: postmark.ServerClient | null = null
  static get postmark() {
    if (!this._postmark)
      this._postmark = new postmark.ServerClient(Config.postmark.apiToken)
    return this._postmark
  }

  private static _firebase: firebase.app.App | null = null
  static get firebaseAuth() {
    if (!this._firebase)
      this._firebase = firebase.initializeApp({
        credential: applicationDefault(),
      })
    return getAuth(this._firebase)
  }

  private static _gcs: Storage | null = null
  static get gcs() {
    if (!this._gcs) this._gcs = new Storage()
    return this._gcs
  }
}
