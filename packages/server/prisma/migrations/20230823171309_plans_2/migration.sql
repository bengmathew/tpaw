-- AlterTable
ALTER TABLE "User" ADD COLUMN     "clientIANATimezoneName" TEXT NOT NULL DEFAULT '',
ADD COLUMN     "nonPlanParams" JSONB,
ADD COLUMN     "nonPlanParamsLastUpdatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP;

-- CreateTable
CREATE TABLE "PlanWithHistory" (
    "userId" TEXT NOT NULL,
    "planId" TEXT NOT NULL,
    "isMain" BOOLEAN NOT NULL,
    "slug" TEXT NOT NULL,
    "label" TEXT,
    "addedToServerAt" TIMESTAMP(3) NOT NULL,
    "sortTime" TIMESTAMP(3) NOT NULL,
    "lastSyncAt" TIMESTAMP(3) NOT NULL,
    "resetCount" INTEGER NOT NULL,
    "endingParams" JSONB NOT NULL,
    "reverseHeadIndex" INTEGER NOT NULL,

    CONSTRAINT "PlanWithHistory_pkey" PRIMARY KEY ("userId","planId")
);

-- CreateTable
CREATE TABLE "PlanParamsChange" (
    "userId" TEXT NOT NULL,
    "planId" TEXT NOT NULL,
    "planParamsChangeId" TEXT NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL,
    "reverseDiff" JSONB NOT NULL,
    "change" JSONB NOT NULL,

    CONSTRAINT "PlanParamsChange_pkey" PRIMARY KEY ("userId","planId","planParamsChangeId")
);

-- CreateIndex
CREATE UNIQUE INDEX "PlanWithHistory_userId_slug_key" ON "PlanWithHistory"("userId", "slug");

-- CreateIndex
CREATE INDEX "PlanParamsChange_userId_planId_timestamp_idx" ON "PlanParamsChange"("userId", "planId", "timestamp" DESC);

-- CreateIndex
CREATE UNIQUE INDEX "PlanParamsChange_userId_planId_timestamp_key" ON "PlanParamsChange"("userId", "planId", "timestamp");

-- AddForeignKey
ALTER TABLE "PlanWithHistory" ADD CONSTRAINT "PlanWithHistory_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PlanParamsChange" ADD CONSTRAINT "PlanParamsChange_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PlanParamsChange" ADD CONSTRAINT "PlanParamsChange_userId_planId_fkey" FOREIGN KEY ("userId", "planId") REFERENCES "PlanWithHistory"("userId", "planId") ON DELETE CASCADE ON UPDATE CASCADE;
