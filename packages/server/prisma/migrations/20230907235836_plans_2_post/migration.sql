/*
  Warnings:

  - You are about to drop the column `planId` on the `User` table. All the data in the column will be lost.
  - You are about to drop the `Plan` table. If the table is not empty, all the data it contains will be lost.
  - Made the column `nonPlanParams` on table `User` required. This step will fail if there are existing NULL values in that column.

*/
-- DropForeignKey
ALTER TABLE "User" DROP CONSTRAINT "User_planId_fkey";

-- DropIndex
DROP INDEX "User_planId_key";

-- AlterTable
ALTER TABLE "User" DROP COLUMN "planId",
ALTER COLUMN "clientIANATimezoneName" DROP DEFAULT,
ALTER COLUMN "nonPlanParams" SET NOT NULL,
ALTER COLUMN "nonPlanParamsLastUpdatedAt" DROP DEFAULT;

-- DropTable
DROP TABLE "Plan";
