import type { DeepMocked } from '@golevelup/ts-jest';
import { createMock } from '@golevelup/ts-jest';
import type { TestingModule } from '@nestjs/testing';
import { Test } from '@nestjs/testing';

import { CommunityConfigService } from '@/config/community';
import { u } from '@/test/factories';

import { EntityEdgeRepository, EntityNodeRepository } from '../repository/repositories';
import { CommunityMaintenanceService } from './community-maintenance.service';
import { CommunityRebuildQueueService, CommunityUpdateQueueService } from './queue';

describe('CommunityMaintenanceService', () => {
  let service: CommunityMaintenanceService;
  let config: {
    rebuildMaxNodes: number;
    rebuildDebounceMs: number;
    rebuildLimitEnabled: boolean;
  };
  let nodeRepo: DeepMocked<EntityNodeRepository>;
  let edgeRepo: DeepMocked<EntityEdgeRepository>;
  let rebuildQueue: DeepMocked<CommunityRebuildQueueService>;
  let updateQueue: DeepMocked<CommunityUpdateQueueService>;

  const userId = u('user-1');
  const graphId = u('graph-1');
  const entityIds = [u('e-1'), u('e-2')];

  beforeEach(async () => {
    config = { rebuildMaxNodes: 0, rebuildDebounceMs: 30000, rebuildLimitEnabled: false };
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        CommunityMaintenanceService,
        { provide: CommunityConfigService, useValue: config },
      ],
    })
      .useMocker(createMock)
      .compile();

    service = module.get(CommunityMaintenanceService);
    nodeRepo = module.get(EntityNodeRepository);
    edgeRepo = module.get(EntityEdgeRepository);
    rebuildQueue = module.get(CommunityRebuildQueueService);
    updateQueue = module.get(CommunityUpdateQueueService);
    nodeRepo.countForGraph.mockResolvedValue(10);
    edgeRepo.countForGraph.mockResolvedValue(20);
  });

  it('schedules a debounced rebuild when the limit is disabled', async () => {
    await service.scheduleMaintenance(userId, graphId, entityIds);

    expect(rebuildQueue.enqueue).toHaveBeenCalledWith({ userId, graphId }, 30000);
    expect(updateQueue.enqueue).not.toHaveBeenCalled();
  });

  it('schedules a rebuild when under the configured limit', async () => {
    config.rebuildLimitEnabled = true;
    config.rebuildMaxNodes = 100;

    await service.scheduleMaintenance(userId, graphId, entityIds);

    expect(rebuildQueue.enqueue).toHaveBeenCalledWith({ userId, graphId }, 30000);
    expect(updateQueue.enqueue).not.toHaveBeenCalled();
  });

  it('falls back to the incremental update path when over the limit', async () => {
    config.rebuildLimitEnabled = true;
    config.rebuildMaxNodes = 5;
    nodeRepo.countForGraph.mockResolvedValue(10);

    await service.scheduleMaintenance(userId, graphId, entityIds);

    expect(updateQueue.enqueue).toHaveBeenCalledWith({ userId, graphId, entityIds });
    expect(rebuildQueue.enqueue).not.toHaveBeenCalled();
  });

  it('does nothing when no entity ids are touched', async () => {
    await service.scheduleMaintenance(userId, graphId, []);

    expect(nodeRepo.countForGraph).not.toHaveBeenCalled();
    expect(rebuildQueue.enqueue).not.toHaveBeenCalled();
    expect(updateQueue.enqueue).not.toHaveBeenCalled();
  });
});
