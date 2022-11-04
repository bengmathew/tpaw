import {
  Block,
  BLOCKS,
  Document,
  helpers,
  Inline,
  INLINES,
  Mark,
  MARKS,
  Text,
  TopLevelBlock,
} from '@contentful/rich-text-types'
import * as contentful from 'contentful'
import _ from 'lodash'
import Link from 'next/link'
import React, {cloneElement, isValidElement, ReactNode} from 'react'
import {Config} from '../Pages/Config'
import {assert, fGet} from './Utils'

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

  export type FetchedInline = {TPAW: Document; SPAW: Document; SWR: Document}
  export const fetchInline = async (id: string) =>
    (await fetchInlineMultiple({x: id})).x

  // Thanks: https://stackoverflow.com/a/47842314
  type _IndirectIdObj<X> = Record<string, string | X>
  // eslint-disable-next-line @typescript-eslint/no-empty-interface
  interface _IdObj extends _IndirectIdObj<_IdObj> {}
  type _MapIdObj<IdObj extends _IdObj> = {
    [P in keyof IdObj]: IdObj[P] extends string
      ? FetchedInline
      : IdObj[P] extends _IdObj
      ? _MapIdObj<IdObj[P]>
      : never
  }
  export const fetchInlineMultiple = async <X extends _IdObj>(
    idObj: X
  ): Promise<_MapIdObj<X>> => {
    const getIdArr = (idObj: _IdObj): string[] => {
      const values = _.values(idObj)
      return _.flatten(
        values.map(x => (typeof x === 'string' ? [x] : getIdArr(x)))
      )
    }
    const idArr = getIdArr(idObj)

    const {items} = await _client().getEntries<InlineEntry>({
      content_type: 'inline',
      'sys.id[in]': idArr.join(','),
    })
    const idMap = new Map<string, FetchedInline>()
    items.forEach(item => {
      const TPAW = item.fields.tpawBody
      const SPAW = item.fields.spawBody ?? TPAW
      const SWR = item.fields.swrBody ?? TPAW
      idMap.set(item.sys.id, {TPAW, SPAW, SWR})
    })

    const getResultObj = <X extends _IdObj>(idObj: X): _MapIdObj<X> =>
      _.mapValues(idObj, x =>
        typeof x === 'string' ? fGet(idMap.get(x)) : getResultObj(x)
      ) as unknown as _MapIdObj<X>

    return getResultObj(idObj)
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

  export const replaceVariables = <T extends Text | Block | Inline>(
    variables: Record<string, string>,
    node: T
  ): T => {
    if (helpers.isText(node)) {
      const keys = _.keys(variables)
      const replace = (x: string) =>
        keys.reduce((a, c) => a.replace(`{{${c}}}`, variables[c]), x)
      return {...node, value: replace(node.value)}
    } else {
      return {
        ...node,
        content: node.content.map((x: any) => replaceVariables(variables, x)),
      }
    }
  }
  export const splitDocument = (
    document: Document,
    marker: string
  ): {
    intro: null | Document
    sections: {heading: string; body: Document}[]
  } => {
    const contentWithMarkerFlag = document.content.map(x =>
      _markerAnalysis(x, marker)
    )
    const takeNodes = () => {
      const result: TopLevelBlock[] = []
      let curr = contentWithMarkerFlag[0]
      while (curr && !curr.isMarker) {
        result.push(curr.node)
        contentWithMarkerFlag.shift()
        curr = contentWithMarkerFlag[0]
      }
      return result
    }
    const introContent = takeNodes()
    let intro =
      introContent.length > 0 ? {...document, content: introContent} : null

    const sections: {heading: string; body: Document}[] = []
    while (contentWithMarkerFlag.length > 0) {
      const first = fGet(contentWithMarkerFlag.shift())
      assert(first.isMarker)
      sections.push({
        heading: first.heading,
        body: {...document, content: takeNodes()},
      })
    }
    assert(sections.every(x => x.body.content.length > 0))
    return {intro, sections}
  }

  type ClassNameSpec = string | ((path: number[]) => string)
  export const RichText = React.memo(
    ({
      body,
      h1 = '',
      h2 = '',
      h3 = '',
      h4 = '',
      h5 = '',
      p6 = '',
      li = '',
      ol = '',
      ul = '',
      p = '',
      a = 'underline',
      bold = 'font-bold',
      italic = 'italic',
    }: {
      body: Document
      h1?: ClassNameSpec
      h2?: ClassNameSpec
      h3?: string
      h4?: string
      h5?: string
      p6?: string
      p?: string
      li?: string
      ul?: string
      ol?: string
      a?: string
      bold?: string
      italic?: string
    }) => {
      const href = (node: Block | Inline) => {
        const {target} = node.data as {
          target: contentful.Entry<StandaloneEntry>
        }

        return `/${target.fields.type}/${target.fields.slug}`
      }
      const linkStyle: React.CSSProperties = {overflowWrap: 'anywhere'}

      const className = (x: ClassNameSpec, path: number[]) =>
        typeof x === 'string' ? x : x(path)
      return (
        <>
          {_nodeToReactComponent(body, {
            nodeRenderers: {
              [BLOCKS.HEADING_1]: (node, children, path) => (
                <h1 className={className(h1, path)}>{children}</h1>
              ),
              [BLOCKS.HEADING_2]: (node, children, path) => (
                <h2 className={className(h2, path)}>{children}</h2>
              ),
              [BLOCKS.HEADING_3]: (node, children) => (
                <h3 className={h3}>{children}</h3>
              ),
              [BLOCKS.HEADING_4]: (node, children) => (
                <h4 className={h4}>{children}</h4>
              ),
              [BLOCKS.HEADING_5]: (node, children) => (
                <h5 className={h5}>{children}</h5>
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
                <ul className={ul}>{children}</ul>
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
            },
            markRenderers: {
              [MARKS.ITALIC]: children => {
                return <span className={italic}>{children}</span>
              },
              [MARKS.BOLD]: children => {
                return <span className={bold}>{children}</span>
              },
            },
          })}
        </>
      )
    }
  )
}
const _markerAnalysis = (
  node: TopLevelBlock,
  marker: string
):
  | {isMarker: false; node: TopLevelBlock}
  | {isMarker: true; heading: string} => {
  const fullMarker = `[[${marker}]]`
  if (node.nodeType !== BLOCKS.HEADING_2) return {isMarker: false, node}
  if (node.content.length === 0) return {isMarker: false, node}
  const firstChild = node.content[0]
  if (!helpers.isText(firstChild)) return {isMarker: false, node}
  if (!firstChild.value.startsWith(fullMarker)) return {isMarker: false, node}
  return {
    isMarker: true,
    heading: firstChild.value.substring(fullMarker.length),
  }
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

type NodeRenderer = (
  node: Block | Inline,
  children: ReactNode,
  path: number[]
) => ReactNode
type MarkRenderer = (children: ReactNode) => ReactNode
type NodeRenderers = Record<string, NodeRenderer>
type MarkRenderers = Record<string, MarkRenderer>
function _nodeToReactComponent(
  node: Text | Block | Inline,
  options: {
    nodeRenderers: NodeRenderers
    markRenderers: MarkRenderers
  },
  path: number[] = []
): ReactNode {
  const {nodeRenderers, markRenderers} = options
  if (helpers.isText(node)) {
    return node.marks.reduce((value: ReactNode, mark: Mark): ReactNode => {
      const renderer = markRenderers?.[mark.type]
      if (!renderer) {
        throw new Error(`No renderer for ${mark.type}`)
      }
      return renderer(value)
    }, node.value)
  } else {
    const children: ReactNode = node.content
      .map((node, index) =>
        _nodeToReactComponent(node, options, [index, ...path])
      )
      .map((element, index) =>
        isValidElement(element) && element.key === null
          ? cloneElement(element, {key: index})
          : element
      )
    if (node.nodeType === 'document') {
      return <>{children}</>
    }
    const renderer = nodeRenderers[node.nodeType]
    if (!renderer) {
      throw new Error(`No renderer for ${node.nodeType}`)
    }
    return renderer(node, children, path)
  }
}
