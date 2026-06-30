import { Body, Controller, Get, Post, UseGuards } from '@nestjs/common';
import { ApiOperation, ApiTags } from '@nestjs/swagger';
import { createZodDto } from 'nestjs-zod';
import { z } from 'zod';

import { GetUser } from '@/auth/decorator';
import { LoggedInGuard } from '@/auth/guard';
import type { Uuid } from '@/common/schemas';
import { CommunityService } from '@/knowledge-graph/community';
import { EpisodeService } from '@/knowledge-graph/episode';
import { Traceable } from '@/observability';
import type { UserWithoutPassword } from '@/user/dto';

import { AgentService } from './agent.service';

// TODO: REMOVE - test DTOs.
const TestIngestSchema = z.object({
  name: z.string().min(1),
  content: z.string().min(1),
});
class TestIngestDto extends createZodDto(TestIngestSchema) {}

const ChatSchema = z.object({
  message: z.string().min(1),
});
class ChatDto extends createZodDto(ChatSchema) {}

@ApiTags('Agent')
@Controller('agent')
@UseGuards(LoggedInGuard)
@Traceable({ asLangfuseTrace: true })
export class AgentController {
  constructor(
    private readonly agentService: AgentService,
    private readonly episodeService: EpisodeService, // TODO: REMOVE
    private readonly communityService: CommunityService, // TODO: REMOVE
  ) {}

  @Get()
  @ApiOperation({ summary: 'Get agent information and configuration' })
  getAgentInfo(@GetUser('id') userId: Uuid) {
    return this.agentService.getAgentInfo(userId);
  }

  // TODO: REMOVE
  @Post('test/ingest')
  @ApiOperation({
    summary: '[TEST] Ingest an episode into the knowledge graph',
  })
  async testIngest(@Body() body: TestIngestDto, @GetUser() user: UserWithoutPassword) {
    const graphId = user.graphs.find((g) => g.name === 'main')!.id;
    const [result] = await this.episodeService.addTextEpisodes({
      userId: user.id,
      episodes: [
        {
          name: body.name,
          content: body.content,
          graphId,
          sourceDescription: 'RememberNow UI',
        },
      ],
      updateCommunities: true,
    });

    return {
      episodeId: result.episode.id,
      nodes: result.nodes.map((n) => ({
        id: n.id,
        name: n.name,
        summary: n.summary,
        labels: n.labels,
      })),
      edges: result.edges.map((e) => ({
        id: e.id,
        name: e.name,
        fact: e.fact,
        validAt: e.validAt?.toISOString() ?? null,
        invalidAt: e.invalidAt?.toISOString() ?? null,
      })),
      invalidatedEdgeCount: result.invalidatedEdges.length,
    };
  }

  @Post('chat')
  @ApiOperation({ summary: 'Chat with the agent (pre-fetch + knowledge search)' })
  async chat(@Body() body: ChatDto, @GetUser() user: UserWithoutPassword) {
    const graphId = user.graphs.find((g) => g.name === 'main')!.id;
    const { text, grounding } = await this.agentService.chat({
      userId: user.id,
      graphIds: [graphId],
      message: body.message,
    });
    return { reply: text, grounding };
  }

  // TODO: REMOVE - test endpoint; replace with proper user-facing button later.
  @Post('test/rebuild-communities')
  @ApiOperation({ summary: '[TEST] Force rebuild communities (bypasses 5min debounce)' })
  async testRebuildCommunities(@GetUser() user: UserWithoutPassword) {
    const graphId = user.graphs.find((g) => g.name === 'main')!.id;
    await this.communityService.buildCommunities(user.id, graphId, {
      userId: user.id,
      tags: ['knowledge-graph', 'community-rebuild', `graph:${graphId}`],
      metadata: { trigger: 'manual' },
    });
    return { ok: true, graphId };
  }
}
