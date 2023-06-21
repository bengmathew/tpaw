import { assert, fGet, noCase } from '@tpaw/common'
import clsx from 'clsx'
import { path } from 'd3-path'
import _ from 'lodash'
import React, { ReactNode, useLayoutEffect, useMemo, useState } from 'react'
import { formatCurrency } from '../../../../Utils/FormatCurrency'

export namespace Sankey {
  export const Chart = React.memo(
    ({
      className,
      model,
      heightOfMaxQuantity,
      sizing: sizingIn,
      vertAlign,
    }: {
      className?: string
      model: Model
      heightOfMaxQuantity: number
      vertAlign: 'top' | 'center'
      sizing: {
        nodeYGap: number
        minNodeXGap: number
        maxNodeXGap: number
        nodeWidth: number
        paddingTop: number
        paddingBottom: number
      }
    }) => {
      const [outerWidth, setOuterWidth] = useState(0)
      const [outerDiv, setOuterDiv] = useState<HTMLElement | null>(null)
      useLayoutEffect(() => {
        if (!outerDiv) return
        const observer = new ResizeObserver((entries) => {
          setOuterWidth(outerDiv.getBoundingClientRect().width)
        })
        observer.observe(outerDiv)
        return () => observer.disconnect()
      }, [outerDiv])

      const { getNodeInfo, sizing } = useMemo(() => {
        const numCols = model.length
        const nodeXGap = _.clamp(
          (outerWidth - sizingIn.nodeWidth * numCols) / (numCols - 1),
          sizingIn.minNodeXGap,
          sizingIn.maxNodeXGap,
        )
        const width = sizingIn.nodeWidth * numCols + nodeXGap * (numCols - 1)
        const colQuantities = model.map((col) =>
          _.sumBy(col.nodes, getNodeQuantity),
        )
        const maxColumnQuantity = fGet(_.max(colQuantities))
        const scale = maxColumnQuantity / heightOfMaxQuantity
        const getColInfo = (col: ColumnModel) => {
          const height =
            _.sumBy(col.nodes, getNodeQuantity) / scale +
            sizingIn.nodeYGap * (col.nodes.length - 1)
          return { height }
        }

        const height =
          Math.max(...model.map((col) => getColInfo(col).height)) +
          sizingIn.paddingTop +
          sizingIn.paddingBottom
        const sizing: _Sizing = {
          width,
          scale,
          height,
          nodeXGap,
          nodeWidth: sizingIn.nodeWidth,
          nodeYGap: sizingIn.nodeYGap,
          paddingTop: sizingIn.paddingTop,
        }

        const getNodeInfo = (node: NodeModel) => {
          const colIndex = fGet(model.findIndex((x) => x.nodes.includes(node)))
          assert(colIndex !== -1)
          const col = model[colIndex]
          const { height: colHeight } = getColInfo(col)
          const rowIndex = col.nodes.indexOf(node)
          assert(rowIndex !== -1)
          const y =
            _.sumBy(col.nodes.slice(0, rowIndex), getNodeQuantity) /
              sizing.scale +
            rowIndex * sizingIn.nodeYGap +
            sizingIn.paddingTop +
            (vertAlign === 'top' ? 0 : (sizing.height - colHeight) / 2)
          const quantity = getNodeQuantity(node)
          const height = Math.max(0.5, quantity / sizing.scale)
          return { y, height, colIndex, quantity }
        }
        return { getNodeInfo, getColInfo, sizing }
      }, [
        heightOfMaxQuantity,
        model,
        outerWidth,
        sizingIn.maxNodeXGap,
        sizingIn.minNodeXGap,
        sizingIn.nodeWidth,
        sizingIn.nodeYGap,
        sizingIn.paddingBottom,
        sizingIn.paddingTop,
        vertAlign,
      ])

      const elements = (() =>
        model.map((col, columnIndex) => {
          return col.nodes.map((node, rowIndex) => {
            if (node.hidden) return null
            const { y, height, quantity } = getNodeInfo(node)
            const lineElements = (() => {
              switch (node.type) {
                case 'start':
                  return null
                case 'merge': {
                  let currY = y
                  return node.srcs.map((src, i) => {
                    assert(!src.hidden)
                    const srcInfo = getNodeInfo(src)
                    const line = (
                      <_Edge
                        key={i}
                        startColumnIndex={srcInfo.colIndex}
                        endColumnIndex={columnIndex}
                        startY={srcInfo.y}
                        endY={currY}
                        height={srcInfo.height}
                        sizing={sizing}
                        color={{
                          src: src.color.edge,
                          dest: node.color.edge,
                        }}
                      />
                    )
                    currY += srcInfo.height
                    return line
                  })
                }
                case 'split':
                  assert(!node.src.hidden)
                  const srcInfo = getNodeInfo(node.src)
                  return (
                    <_Edge
                      startColumnIndex={srcInfo.colIndex}
                      endColumnIndex={columnIndex}
                      startY={srcInfo.y + node.offset / sizing.scale}
                      endY={y}
                      height={height}
                      sizing={sizing}
                      color={{
                        src: node.src.color.edge,
                        dest: node.color.edge,
                      }}
                    />
                  )

                default:
                  noCase(node)
              }
            })()

            const nodeElement = (
              <_Node
                key={`${columnIndex}-${rowIndex}`}
                columnIndex={columnIndex}
                y={y}
                quantity={quantity}
                label={node.label}
                height={height}
                labelPosition={col.labelPosition}
                sizing={sizing}
                color={node.color.node ?? 'black'}
              />
            )
            return { lineElements, nodeElement }
          })
        }))()
      return (
        <div
          className={clsx(className, ' overflow-x-auto ')}
          style={{ height: `${sizing.height}px` }}
          ref={setOuterDiv}
        >
          <div
            className="flex-shrink-0"
            style={{ width: `${sizing.width}px`, height: `${sizing.height}px` }}
          >
            <svg
              className="w-full h-full"
              viewBox={`0 0 ${sizing.width} ${sizing.height}`}
            >
              {elements.map((col) => col.map((x) => x?.lineElements))}
              {elements.map((col) => col.map((x) => x?.nodeElement))}
            </svg>
          </div>
        </div>
      )
    },
  )

  type _Color = {
    node: string
    edge: string
  }
  export type NodeModel = {
    label: ReactNode
    color: _Color
    hidden: boolean
  } & (
    | { type: 'start'; quantity: number }
    | { type: 'split'; quantity: number; offset: number; src: NodeModel }
    | { type: 'merge'; srcs: NodeModel[] }
  )

  export type ColumnModel = {
    labelPosition: 'left' | 'right' | 'top'
    nodes: NodeModel[]
  }

  export type Model = ColumnModel[]

  const getNodeQuantity = (node: NodeModel): number => {
    switch (node.type) {
      case 'split':
      case 'start':
        return node.quantity
      case 'merge':
        return _.sumBy(node.srcs, getNodeQuantity)
      default:
        noCase(node)
    }
  }

  export const splitNode = (
    src: NodeModel,
    parts: ({
      quantity: number
      label: ReactNode
      color: _Color
      hidden: boolean
    } | null)[],
  ) => {
    let offset = 0
    return parts.map((part) => {
      if (!part) return null
      const node: Extract<NodeModel, { type: 'split' }> = {
        type: 'split',
        label: part.label,
        quantity: part.quantity,
        color: part.color,
        hidden: part.hidden,
        offset,
        src,
      }
      offset += part.quantity
      return node
    })
  }

  type _Sizing = {
    width: number
    height: number
    scale: number
    nodeXGap: number
    nodeWidth: number
    nodeYGap: number
    paddingTop: number
  }

  const _Node = React.memo(
    ({
      className,
      columnIndex,
      y,
      quantity,
      label,
      labelPosition,
      sizing,
      height,
      color,
    }: {
      className?: string
      columnIndex: number
      y: number
      quantity: number
      height:number
      label: ReactNode
      labelPosition: 'left' | 'right' | 'top'
      sizing: _Sizing
      color: string
    }) => {
      const x = columnIndex * (sizing.nodeXGap + sizing.nodeWidth)
      const labelWidth =
        labelPosition === 'top'
          ? sizing.width
          : sizing.nodeXGap - sizing.nodeWidth - 5
      return (
        <>
          <rect
            className={clsx(className)}
            fill={color}
            x={x}
            y={y}
            width={sizing.nodeWidth}
            height={height}
            rx={2}
          />
          <foreignObject
            x={
              labelPosition === 'left'
                ? x + sizing.nodeWidth + 2
                : labelPosition === 'right'
                ? x - labelWidth - 2
                : labelPosition === 'top'
                ? x + sizing.nodeWidth/2 - labelWidth / 2
                : noCase(labelPosition)
            }
            y={
              labelPosition === 'left' || labelPosition === 'right'
                ? y - sizing.nodeYGap / 2
                : labelPosition === 'top'
                ? y - sizing.paddingTop
                : noCase(labelPosition)
            }
            width={labelWidth}
            height={height + sizing.nodeYGap}
          >
            <div
              className={clsx(
                ' h-full flex flex-col',
                labelPosition === 'right'
                  ? 'justify-center  items-end text-right'
                  : labelPosition === 'left'
                  ? 'justify-center  items-start'
                  : labelPosition === 'top'
                  ? 'justify-start  items-center'
                  : noCase(labelPosition),
              )}
            >
              {label}
              {labelPosition !== 'top' && (
                <h2 className="text-[11px] lighten -mt-0.5">
                  {formatCurrency(quantity)}
                </h2>
              )}
            </div>
          </foreignObject>
        </>
      )
    },
  )

  const _Edge = React.memo(
    ({
      className,
      startColumnIndex,
      endColumnIndex,
      startY,
      endY,
      height,
      sizing,
      color,
    }: {
      className?: string
      startColumnIndex: number
      endColumnIndex: number
      startY: number
      endY: number
      height: number
      color: { src: string; dest: string }
      sizing: _Sizing
    }) => {
      const gradientId = _.uniqueId()
      const start = {
        x:
          startColumnIndex * (sizing.nodeXGap + sizing.nodeWidth) +
          sizing.nodeWidth +
          0,
        y: startY,
        bottom: startY + height,
      }
      const end = {
        x: endColumnIndex * (sizing.nodeXGap + sizing.nodeWidth) - 0,
        y: endY,
        bottom: endY + height,
      }
      return (
        <>
          <defs>
            <linearGradient id={gradientId}>
              <stop offset="0%" stop-color={color.src} />
              <stop offset="100%" stop-color={color.dest} />
            </linearGradient>
          </defs>
          <path
            className={clsx(className)}
            stroke={'none'}
            strokeWidth={height}
            fill={`url(#${gradientId})`}
            d={(() => {
              const p = path()

              const bx1 = start.x + sizing.nodeXGap * 0.50
              p.moveTo(start.x, start.y)
              const bx2 = end.x - sizing.nodeXGap * 0.50
              p.bezierCurveTo(bx1, start.y, bx2, end.y, end.x, end.y)
              p.lineTo(end.x, end.bottom)
              p.bezierCurveTo(
                bx2,
                end.bottom,
                bx1,
                start.bottom,
                start.x,
                start.y + height,
              )

              return p.toString()
            })()}
          />
        </>
      )
    },
  )
}
