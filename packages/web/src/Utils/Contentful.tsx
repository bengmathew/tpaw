import {documentToReactComponents} from '@contentful/rich-text-react-renderer'
import {
  Block,
  BLOCKS,
  Document,
  Inline,
  INLINES,
} from '@contentful/rich-text-types'
import * as contentful from 'contentful'
import _ from 'lodash'
import Link from 'next/link'
import React from 'react'
import {Config} from '../Pages/Config'
import {assert, assertFalse, fGet} from './Utils'

export namespace Contentful {
  export type InlineEntry = {
    title: string
    tpawBody: Document
    spawBody?: Document
    swrBody?: Document
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
    const item = fGet(items[0])
    const TPAW = item.fields.tpawBody
    const SPAW = item.fields.spawBody ?? TPAW
    const SWR = item.fields.swrBody ?? TPAW
    return {TPAW, SPAW, SWR}
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

  // THIS IS A HACK.
  export const replaceVariables = (
    variables: Record<string, string>,
    node: Document
  ): Document => {
    const nodeAny = node as any
    if ('content' in nodeAny) {
      return {
        ...node,
        content: nodeAny.content.map((x: any) =>
          // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
          replaceVariables(variables, x)
        ),
      }
    } else if ('value' in nodeAny) {
      const keys = _.keys(variables)
      const replace = (x: string) =>
        keys.reduce((a, c) => a.replace(`{{${c}}}`, variables[c]), x)
      const result: any = {
        ...node,
        // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
        value: replace(nodeAny.value),
      }
      return result
    } else {
      assertFalse()
    }
  }

  export const RichText = React.memo(
    ({
      body,
      h1 = '',
      h2 = '',
      li = '',
      ol = '',
      ul = '',
      p = '',
      p6 = '',
      a = 'underline ',
      aExternalLink = 'text-[12px] ml-1',
      bold = 'font-bold',
    }: {
      body: Document
      h1?: string
      h2?: string
      p?: string
      p6?: string
      li?: string
      ul?: string
      ol?: string
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
      const linkStyle: React.CSSProperties = {overflowWrap: 'anywhere'}

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
              [BLOCKS.HEADING_6]: (node, children) => (
                <p className={p6}>{children}</p>
              ),
              [BLOCKS.PARAGRAPH]: (node, children) => (
                <p className={p}>{children}</p>
              ),
              [BLOCKS.LIST_ITEM]: (node, children) => (
                <li className={li}>{children}</li>
              ),
              [BLOCKS.UL_LIST]: (node, children) => (
                <ol className={ul}>{children}</ol>
              ),
              [BLOCKS.OL_LIST]: (node, children) => (
                <ol className={ol}>{children}</ol>
              ),
              [INLINES.HYPERLINK]: (node, children) => {
                const href = node.data.uri as string
                return href.startsWith('/') ? (
                  <Link href={href}>
                    <a className={a} style={linkStyle}>
                      {children}
                    </a>
                  </Link>
                ) : (
                  <a
                    className={a}
                    href={href}
                    target="_blank"
                    rel="noreferrer"
                    style={linkStyle}
                  >
                    {children}
                  </a>
                )
              },
              [INLINES.ENTRY_HYPERLINK]: (node, children) => (
                <Link href={href(node)}>
                  <a className={`${a}`} style={linkStyle}>
                    {children}
                  </a>
                </Link>
              ),
              // This does not seem to work!
              // [MARKS.BOLD]: (node, children) => {
              //   return <span className={bold}>{children}</span>
              // },
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
