/*
  Warnings:

  - You are about to drop the `has_episode_edges` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `saga_nodes` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "has_episode_edges" DROP CONSTRAINT "has_episode_edges_episodic_id_fkey";

-- DropForeignKey
ALTER TABLE "has_episode_edges" DROP CONSTRAINT "has_episode_edges_graph_id_fkey";

-- DropForeignKey
ALTER TABLE "has_episode_edges" DROP CONSTRAINT "has_episode_edges_saga_id_fkey";

-- DropForeignKey
ALTER TABLE "saga_nodes" DROP CONSTRAINT "saga_nodes_graph_id_fkey";

-- AlterTable
ALTER TABLE "episodic_nodes" ADD COLUMN     "saga_id" UUID;

-- DropTable
DROP TABLE "has_episode_edges";

-- DropTable
DROP TABLE "saga_nodes";

-- CreateTable
CREATE TABLE "sagas" (
    "id" UUID NOT NULL,
    "graph_id" UUID NOT NULL,
    "name" TEXT NOT NULL,
    "labels" TEXT[],
    "summary" TEXT NOT NULL DEFAULT '',
    "last_summarized_at" TIMESTAMPTZ(3),
    "created_at" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "sagas_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "sagas_graph_id_idx" ON "sagas"("graph_id");

-- CreateIndex
CREATE INDEX "sagas_name_idx" ON "sagas"("name");

-- CreateIndex
CREATE INDEX "episodic_nodes_saga_id_idx" ON "episodic_nodes"("saga_id");

-- AddForeignKey
ALTER TABLE "episodic_nodes" ADD CONSTRAINT "episodic_nodes_saga_id_fkey" FOREIGN KEY ("saga_id") REFERENCES "sagas"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "sagas" ADD CONSTRAINT "sagas_graph_id_fkey" FOREIGN KEY ("graph_id") REFERENCES "graphs"("id") ON DELETE CASCADE ON UPDATE CASCADE;
