/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import AddObjectButton from '@/common/components/annotations/AddObjectButton';
import FirstClickView from '@/common/components/annotations/FirstClickView';
import ObjectsToolbarBottomActions from '@/common/components/annotations/ObjectsToolbarBottomActions';
import ObjectsToolbarHeader from '@/common/components/annotations/ObjectsToolbarHeader';
import {getObjectLabel} from '@/common/components/annotations/ObjectUtils';
import ToolbarObject from '@/common/components/annotations/ToolbarObject';
import {
  activeTrackletObjectAtom,
  activeTrackletObjectIdAtom,
  isAddObjectEnabledAtom,
  isFirstClickMadeAtom,
  trackletObjectsAtom,
} from '@/demo/atoms';
import {BaseTracklet} from '@/common/tracker/Tracker';
import {useAtomValue, useSetAtom} from 'jotai';

type Props = {
  onTabChange: (newIndex: number) => void;
};

export default function ObjectsToolbar({onTabChange}: Props) {
  const tracklets = useAtomValue(trackletObjectsAtom);
  const activeTracklet = useAtomValue(activeTrackletObjectAtom);
  const setActiveTrackletId = useSetAtom(activeTrackletObjectIdAtom);
  const isFirstClickMade = useAtomValue(isFirstClickMadeAtom);
  const isAddObjectEnabled = useAtomValue(isAddObjectEnabledAtom);

  if (!isFirstClickMade) {
    return <FirstClickView />;
  }

  return (
    <div className="flex flex-col h-full">
      <ObjectsToolbarHeader />
      <div className="grow w-full overflow-y-auto">
        {tracklets.map((tracklet: BaseTracklet) => {
          return (
            <ToolbarObject
              key={tracklet.id}
              label={getObjectLabel(tracklet)}
              tracklet={tracklet}
              isActive={activeTracklet?.id === tracklet.id}
              onClick={() => {
                setActiveTrackletId(tracklet.id);
              }}
            />
          );
        })}
        {isAddObjectEnabled && <AddObjectButton />}
      </div>
      <ObjectsToolbarBottomActions onTabChange={onTabChange} />
    </div>
  );
}