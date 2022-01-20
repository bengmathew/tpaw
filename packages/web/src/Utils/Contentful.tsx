import {documentToReactComponents} from '@contentful/rich-text-react-renderer'
import {
  Block,
  BLOCKS,
  Document,
  Inline,
  INLINES,
  MARKS,
} from '@contentful/rich-text-types'
import {faExternalLink} from '@fortawesome/pro-light-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import * as contentful from 'contentful'
import Link from 'next/link'
import React from 'react'
import {Config} from '../Pages/Config'
import {assert, fGet} from './Utils'

export namespace Contentful {
  export type InlineEntry = {
    title: string
    body: Document
  }
  export type StandaloneEntry = {
    title: string
    slug: string
    body: Document
    type: 'article'
  }
  export type KnowledgeBaseOutlineEntry = {
    title: string
    items: (
      | contentful.Entry<KnowledgeBaseOutlineEntry>
      | contentful.Entry<KnowledgeBaseArticleEntry>
    )[]
  }
  export type KnowledgeBaseArticleEntry = {
    internalTitle: string
    publicTitle: string
    slug: string
    body: Document
  }

  export type FetchedInline = Awaited<ReturnType<typeof Contentful.fetchInline>>
  export const fetchInline = async (id: string) => {
    const {items} = await _client().getEntries<InlineEntry>({
      content_type: 'inline',
      'sys.id': id,
    })
    return fGet(items[0])
  }

  export type FetchedStandalone = Awaited<
    ReturnType<typeof Contentful.fetchStandalone>
  >
  export const fetchStandalone = async (slug: string) => {
    const {items} = await _client().getEntries<StandaloneEntry>({
      content_type: 'standalone',
      'fields.slug': slug,
    })
    return fGet(items[0])
  }

  export type FetchedKnowledgeBaseOutline = Awaited<
    ReturnType<typeof Contentful.fetchKnowledgeBaseOutline>
  >
  export const fetchKnowledgeBaseOutline = async (id: string) => {
    const {items} = await _client().getEntries<KnowledgeBaseOutlineEntry>({
      content_type: 'knowledgeBaseOutline',
      'sys.id': id,
      include: 10,
    })
    return fGet(items[0])
  }
  export type FetchedKnowledgeBaseArticle = Awaited<
    ReturnType<typeof Contentful.fetchKnowledgeBaseArticle>
  >
  export const fetchKnowledgeBaseArticle = async (slug: string) => {
    const {items} = await _client().getEntries<KnowledgeBaseArticleEntry>({
      content_type: 'knowledgeBaseArticle',
      'fields.slug': slug,
    })
    return fGet(items[0])
  }

  export const getArticleSlugs = async () => {
    const query = `
    query{
      standaloneCollection(limit:10000) {
        total
        items {
          slug
        }
      }
    }`
    const {items, total} = (
      (await _graphql(query, {})) as {
        data: {
          standaloneCollection: {total: number; items: {slug: string}[]}
        }
      }
    ).data.standaloneCollection
    assert(total === items.length)
    return items.map(x => x.slug)
  }

  export const getKnowledgeBaseArticleSlugs = async () => {
    const query = `
    query{
      knowledgeBaseArticleCollection(limit:10000) {
        total
        items {
          slug
        }
      }
    }`
    const {items, total} = (
      (await _graphql(query, {})) as {
        data: {
          knowledgeBaseArticleCollection: {
            total: number
            items: {slug: string}[]
          }
        }
      }
    ).data.knowledgeBaseArticleCollection
    assert(total === items.length)
    return items.map(x => x.slug)
  }

  export const RichText = React.memo(
    ({
      body,
      h1 = '',
      h2 = '',
      li = '',
      p = '',
      a = 'underline',
      aExternalLink = 'text-[12px] ml-1',
      bold = 'font-bold',
    }: {
      body: Document
      h1?: string
      h2?: string
      p?: string
      li?: string
      a?: string
      aExternalLink?: string
      bold?: string
    }) => {
      const href = (node: Block | Inline) => {
        const {target} = node.data as {
          target: contentful.Entry<StandaloneEntry>
        }

        return `/${target.fields.type}/${target.fields.slug}`
      }

      return (
        <>
          {documentToReactComponents(body, {
            renderNode: {
              [BLOCKS.HEADING_1]: (node, children) => (
                <h1 className={h1}>{children}</h1>
              ),
              [BLOCKS.HEADING_2]: (node, children) => (
                <h2 className={h2}>{children}</h2>
              ),
              [BLOCKS.PARAGRAPH]: (node, children) => (
                <p className={p}>{children}</p>
              ),
              [BLOCKS.LIST_ITEM]: (node, children) => (
                <li className={li}>{children}</li>
              ),
              [INLINES.HYPERLINK]: (node, children) => {
                const href = node.data.uri as string
                return href.startsWith('/') ? (
                  <Link href={href}>
                    <a className={a}>{children}</a>
                  </Link>
                ) : (
                  <a className={a} href={href} target="_blank" rel="noreferrer">
                    {children}
                    {/* <FontAwesomeIcon
                      className={aExternalLink}
                      icon={faExternalLink}
                    /> */}
                  </a>
                )
              },
              [INLINES.ENTRY_HYPERLINK]: (node, children) => (
                <Link href={href(node)}>
                  <a className={a}>{children}</a>
                </Link>
              ),
              [MARKS.BOLD]: (node, children) => (
                <span className={bold}>{children}</span>
              ),
            },
          })}
        </>
      )
    }
  )
}

const _client = () =>
  contentful.createClient({
    ...Config.server.contentful,
    environment: 'master',
  })

const _graphql = async (query: string, variables: any) => {
  const response = await fetch(
    `https://graphql.contentful.com/content/v1/spaces/${Config.server.contentful.space}`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${Config.server.contentful.accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({query, variables}),
    }
  )
  const result = await response.json()
  return result
}
