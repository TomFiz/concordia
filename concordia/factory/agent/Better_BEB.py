# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An Agent Factory."""

from collections.abc import Callable
import datetime
import json

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component
from concordia.utils import measurements as measurements_lib
import numpy as np


DEFAULT_GOAL_COMPONENT_NAME = 'Goal'


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__



# ~~~~ CUSTOM QUESTIONS ~~~~

class AvailableOptionsPerception(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers the question 'what actions are available to me?'."""

  def __init__(self, **kwargs):

    super().__init__(
        question=(
            "Looking at {agent_name}'s goal and biases, what actions are available to "
            "{agent_name} right now? \nIf the protagonist's last observations "
            ' is a call to a final action rather than a simple discussion, then : '
            '\n1) All of the available options should be concrete actions, '
            'or potential outcomes of the situation, and {agent_name} absolutely cannot '
            'try communicating. \n2) If a consensus has been reached in the last '
            'observation, only consider the options that are in line with this consensus. '
            '\nIf the status of the conversation highlighted '
            'by the latest observations is a simple discussion, then the available '
            "options should be negotiation strategies towards {agent_name}'s goal.\n"
            'Explore diverse opportunities, and provide '
            'a clear explanation of the potential benefits each strategy could '
            'have. You will make this explanation as concise as possible, in '
            'order for an external reader to make an educated decision. For each'
            ' option, provide a thorough anticipation of the other protagonists’s '
            'most likely reactions, and how they might influence the '
            "outcome regarding {agent_name}'s goal."

        ),
        terminators=('\n\n',),
        answer_prefix='',
        add_to_memory=False,
        **kwargs,
    )


class BestOptionPerception(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "Given the statements above, which action would {agent_name} be "
            "most likely to choose given its biases in order to acheive "
            "{agent_name}'s overarching goal ? You will strongly consider the "
            "importance of other protagonists' actions to this end."
        ),
        answer_prefix="{agent_name}'s best course of action is ",
        add_to_memory=False,
        **kwargs,
    )


biases = {
"Loss Aversion" : "{agent_name} is subject to loss aversion, which influences their decision in the following way: {agent_name} tends to prioritize avoiding losses over acquiring equivalent gains, making them more likely to choose safer options even if they reduce potential benefits.",
"Anchoring Bias" : "{agent_name} is subject to anchoring bias, which influences their decision in the following way: {agent_name} places disproportionate weight on the first piece of information they encounter, causing initial values or ideas to heavily frame and influence their subsequent choices.",
"Confirmation Bias" : "{agent_name} is subject to confirmation bias, which influences their decision in the following way: {agent_name} is inclined to seek out or interpret information that supports their pre-existing beliefs, potentially disregarding critical evidence that could lead to a more balanced decision.",
"Sunk Cost Fallacy" : "{agent_name} is subject to the sunk cost fallacy, which influences their decision in the following way: {agent_name} feels compelled to continue with a chosen course of action because of prior investments, even if the current path is no longer the best option to achieve their goals."}


# ~~~~~~ AGENT BUILD ~~~~~~


def build_agent(
    *,
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Agent calls update every time this interval passes.

  Returns:
    An agent.
  """
  del update_time_interval
  if not config.extras.get('main_character', False):
    raise ValueError('This function is meant for a main character '
                     'but it was called on a supporting character.')

  agent_name = config.name

  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)

  measurements = measurements_lib.Measurements()
  instructions = agent_components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
  )

  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )

  observation_label = '\nObservation'
  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )
  observation_summary_label = 'Summary of recent observations'
  observation_summary = agent_components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=24),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel('ObservationSummary').on_next,
  )

  relevant_memories_label = '\nRecalled memories and observations'
  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      model=model,
      components={
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(time_display): 'The current date/time is'},
      num_memories_to_retrieve=5,
      pre_act_key=relevant_memories_label,
      logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
  )

  if config.goal:
    goal_label = '\nOverarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
  else:
    goal_label = None
    overarching_goal = None

  bias_label = '\nBiases'
  chosen_biases = ['Loss Aversion','Anchoring Bias']
  chosen_biases = "".join([f'{biases[bias_name]}\n' for bias_name in chosen_biases])
  bias = agent_components.constant.Constant(
      state=chosen_biases.format(agent_name=agent_name),
      pre_act_key=bias_label,
      logging_channel=measurements.get_channel('Biases').on_next,
  )

  options_perception_components = {}
  options_perception_components.update({
      _get_class_name(bias): bias_label,
      _get_class_name(relevant_memories): relevant_memories_label,
  })
  if config.goal:
    options_perception_components[DEFAULT_GOAL_COMPONENT_NAME] = goal_label
  options_perception_label = (
      f'\nQuestion: Which options are available to {agent_name} '
      'right now?\nAnswer')
  options_perception = AvailableOptionsPerception(
          model=model,
          components=options_perception_components,
          clock_now=clock.now,
          pre_act_key=options_perception_label,
          logging_channel=measurements.get_channel(
              'AvailableOptionsPerception'
          ).on_next,
          num_memories_to_retrieve = 5,
      )

  best_option_perception_label = (
      f'\nQuestion: Of the options available to {agent_name}, and '
      'given their goal, which choice of action or strategy is '
      f'best for {agent_name} to take right now?\nAnswer')
  best_option_perception_components = {}
  if config.goal:
    best_option_perception_components[DEFAULT_GOAL_COMPONENT_NAME] = goal_label
  best_option_perception_components.update({
      _get_class_name(bias) : bias_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      _get_class_name(options_perception): options_perception_label,
  })
  best_option_perception = BestOptionPerception(
          model=model,
          components=best_option_perception_components,
          clock_now=clock.now,
          pre_act_key=best_option_perception_label,
          logging_channel=measurements.get_channel(
              'BestOptionPerception'
          ).on_next,
          num_memories_to_retrieve = 5,
      )

  entity_components = (
      # Components that provide pre_act context.
      instructions,
      bias,
      time_display,
      observation,
      observation_summary,
      relevant_memories,
      options_perception,
      best_option_perception,
  )
  components_of_agent = {
      _get_class_name(component): component for component in entity_components
  }
  components_of_agent[
      agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME
  ] = agent_components.memory_component.MemoryComponent(raw_memory)

  component_order = list(components_of_agent.keys())
  if overarching_goal is not None:
    components_of_agent[DEFAULT_GOAL_COMPONENT_NAME] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, DEFAULT_GOAL_COMPONENT_NAME)

  act_component = agent_components.concat_act_component.ConcatActComponent(
      model=model,
      clock=clock,
      component_order=component_order,
      logging_channel=measurements.get_channel('ActComponent').on_next,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
      config=config,
  )

  return agent


def save_to_json(
    agent: entity_agent_with_logging.EntityAgentWithLogging,
) -> str:
  """Saves an agent to JSON data.

  This function saves the agent's state to a JSON string, which can be loaded
  afterwards with `rebuild_from_json`. The JSON data
  includes the state of the agent's context components, act component, memory,
  agent name and the initial config. The clock, model and embedder are not
  saved and will have to be provided when the agent is rebuilt. The agent must
  be in the `READY` phase to be saved.

  Args:
    agent: The agent to save.

  Returns:
    A JSON string representing the agent's state.

  Raises:
    ValueError: If the agent is not in the READY phase.
  """

  if agent.get_phase() != entity_component.Phase.READY:
    raise ValueError('The agent must be in the `READY` phase to be saved.')

  data = {
      component_name: agent.get_component(component_name).get_state()
      for component_name in agent.get_all_context_components()
  }

  data['act_component'] = agent.get_act_component().get_state()

  config = agent.get_config()
  if config is not None:
    data['agent_config'] = config.to_dict()

  return json.dumps(data)


def rebuild_from_json(
    json_data: str,
    model: language_model.LanguageModel,
    clock: game_clock.MultiIntervalClock,
    embedder: Callable[[str], np.ndarray],
    memory_importance: Callable[[str], float] | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Rebuilds an agent from JSON data."""

  data = json.loads(json_data)

  new_agent_memory = associative_memory.AssociativeMemory(
      sentence_embedder=embedder,
      importance=memory_importance,
      clock=clock.now,
      clock_step_size=clock.get_step_size(),
  )

  if 'agent_config' not in data:
    raise ValueError('The JSON data does not contain the agent config.')
  agent_config = formative_memories.AgentConfig.from_dict(
      data.pop('agent_config')
  )

  agent = build_agent(
      config=agent_config,
      model=model,
      memory=new_agent_memory,
      clock=clock,
  )

  for component_name in agent.get_all_context_components():
    agent.get_component(component_name).set_state(data.pop(component_name))

  agent.get_act_component().set_state(data.pop('act_component'))

  assert not data, f'Unused data {sorted(data)}'
  return agent
