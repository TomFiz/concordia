# @title Imports for agent building
import datetime
import numpy as np  # For calculating average and std deviation
import statistics
from typing import List, Dict, Tuple, Union, Callable

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.components.agent import memory_component
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.typing import entity_component
from concordia.components.agent import action_spec_ignored
# from concordia.typing import entity as entity_lib
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib
from concordia.components.agent import question_of_recent_memories
from concordia.typing import logging


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


### ~~~~ TRUSTEE CIRCLE ~~~~ ###


class TrusteeCircle:
  """Class to manage the trustee circle of cooperative agents."""

  def __init__(
      self,
      # agent_name: str,
  ):
    # Dictionary to track individual agent cooperation scores over time
    self.trusted_agents_scores: Dict[str, List[int]] = {}
    # Minimum average cooperation score to stay in the circle
    self.min_avg_cooperation = 3
    # Maximum std deviation to prevent highly fluctuating behavior
    self.max_cooperation_std = 2.0

  def update_trustee(self, ext_agent_name: str, cooperation_score: int):
    """Add cooperation score for an agent and update their cooperation history."""
    if ext_agent_name not in self.trusted_agents_scores:
      self.trusted_agents_scores[ext_agent_name] = []

    self.trusted_agents_scores[ext_agent_name].append(cooperation_score)
    # print(
    #     f"Agent '{ext_agent_name}' updated with cooperation score"
    #     f' {cooperation_score}.'
    # )

  def calculate_agent_stats(self, ext_agent_name: str) -> Tuple[int, float]:
    """Calculate the average and std deviation of cooperation scores for a specific agent."""
    scores = self.trusted_agents_scores[ext_agent_name]
    if len(scores) > 0:
      avg_score = statistics.harmonic_mean(scores)
      std_score = np.std(scores)
      return avg_score, std_score
    return 3, 0.0  # Return default values if no scores are available

  def check_membership(self, ext_agent_name: str) -> bool:
    """Check if an agent should stay in the trustee circle based on their cooperation history."""
    avg_score, std_score = self.calculate_agent_stats(ext_agent_name)
    if (
        avg_score >= self.min_avg_cooperation
        and std_score <= self.max_cooperation_std
    ):
      return True
    else:
      # Remove agent from circle if they do not meet the criteria
      # print(f"Agent '{ext_agent_name}' removed from trustee circle.")
      return False

  def get_trusted_agents(self) -> List[str]:
    """Return the list of current trusted agents."""
    return [
        agent_name
        for agent_name in self.trusted_agents_scores
        if self.check_membership(agent_name)
    ]

  def get_state(self) -> Dict[str, List[float]]:
    """Return a snapshot of the current trustee circle, with cooperation scores."""
    return self.trusted_agents_scores.copy()


DEFAULT_OBSERVATION_TRUSTEES_PRE_ACT_KEY = 'Strategy'


class ObservationTrust(action_spec_ignored.ActionSpecIgnored):
  """Component that updates the trustee circle based on observations."""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      clock_now: Callable[[], datetime.datetime],
      timeframe: datetime.timedelta,
      memory_component_name: str = memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
      pre_act_key: str = DEFAULT_OBSERVATION_TRUSTEES_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
      agent_name: str,
  ):
    """Initializes the component.

    Args:
        model: Language model to evaluate cooperation scores.
        clock_now: Function that returns the current time.
        timeframe: Delta from current moment to display observations from.
        memory_component_name: Name of the memory component to add observations to.
        pre_act_key: Prefix to add to the output of the component when called in `pre_act`.
        logging_channel: The channel to use for debug logging.
        trustee_circle: The trustee circle object to update.
        agent_name: The name of the agent.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._clock_now = clock_now
    self._timeframe = timeframe
    self._memory_component_name = memory_component_name
    self._logging_channel = logging_channel
    self._trustee_circle = TrusteeCircle()
    self._agent_name = agent_name

  def pre_observe(self, observation: str) -> str:
    memory = self.get_entity().get_component(
        self._memory_component_name, type_=memory_component.MemoryComponent
    )
    memory.add(
        f'[observation] {observation}',
        metadata={'tags': ['observation']},
    )
    return ''

  def _make_pre_act_value(self) -> str:
    """Returns the latest observations and updates the trustee circle."""
    memory = self.get_entity().get_component(
        self._memory_component_name, type_=memory_component.MemoryComponent
    )
    interval_scorer = legacy_associative_memory.RetrieveTimeInterval(
        time_from=self._clock_now() - self._timeframe,
        time_until=self._clock_now(),
        add_time=True,
    )
    mems = memory.retrieve(scoring_fn=interval_scorer)
    mems = [mem.text for mem in mems if '[observation]' in mem.text]
    observations = '\n'.join(mems) + '\n'

    # Update the trustee circle
    self._update_circle(observations)

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Trusted agents': self._trustee_circle.get_trusted_agents(),
        'Circle state': self._trustee_circle.get_state(),
    })

    trusted_agents = ', '.join(self._trustee_circle.get_trusted_agents())
    if trusted_agents == '':
      return (
          f'{self._agent_name} cannot trust anyone at the moment, and will'
          ' choose the action that will most quickly and most surely acheive'
          ' their goal.'
      )

    else:
      return (
          f'{self._agent_name} trusts fully the following persons :'
          f' {trusted_agents}. {self._agent_name} is fully willing to make some'
          ' personal concessions to cooperate with them in order to reach a'
          ' bigger collective objective, while trying to reach their own'
          ' personal goal.'
      )

  def _update_circle(self, observations: str):
    """Check cooperation with the trustee circle and update trust for each agent involved."""

    def calculate_cooperation_score(
        observations: str, model
    ) -> Union[Dict[str, float], List[str]]:
      """Use the language model to evaluate how cooperative each agent in the interaction was."""
      # Modify the prompt to ask for a cooperation score for each agent
      prompt = (
          'Rate the level of cooperation of each agent involved in the'
          ' following interaction on an int scale from 1 to 5.Any agent having'
          ' a score of either 4 or 5 can be considered trustworthy, and is'
          ' willing to make concessions to reach a bigger objective.Agents'
          ' having a score of either 1 or 2 are not necessarly mean, but act'
          ' selfishly, and reluctant to making concessions.If the discussion'
          " contains no indication as of the agent's will to collaborate"
          ' (factual statement demonstrating no choice from the indivual for'
          ' example), you will return 3.Provide a dictionary whith each agent,'
          ' strictly following this format, without justification:'
          f" {{'agent_name': int}}.Here is the interaction : '{observations}'."
      )

      response = model.sample_text(prompt=prompt, temperature=0.0)
      # print(f'Trust LLM response : \n {response}  \n')

      try:
        res_dict = eval(response)  # Convert the model response to a dictionary
        if isinstance(res_dict, dict):
          # print(f'Cooperation score calculation : {res_dict}  \n')
          return res_dict  # Return a dictionary of agent names and their respective cooperation scores
        else:
          return {}
      except (SyntaxError, TypeError):
        return {}

    def store_interaction_memory(
        agent_name: str,
        cooperation_scores: Dict[str, float],
        trustee_circle: TrusteeCircle,
    ):
      """Store the interaction details, including individual cooperation scores and trustee circle state."""
      scores = ', '.join(
          [f'{agent}: {score}' for agent, score in cooperation_scores.items()]
      )

      with open('trust_circle_memory.csv', 'a', encoding='utf-8') as f:
        f.write(
            f'{agent_name}; {scores}; {trustee_circle.get_state()};'
            f' {trustee_circle.get_trusted_agents()}\n'
        )

    # Get the individual cooperation scores for each agent
    cooperation_scores = calculate_cooperation_score(
        observations=observations, model=self._model
    )

    # Update trustee circle for each agent based on their individual cooperation score
    for ext_agent_name, cooperation_score in cooperation_scores.items():
      if ext_agent_name != self._agent_name:
        self._trustee_circle.update_trustee(ext_agent_name, cooperation_score)

    # Store this interaction and trustee circle state in memory
    store_interaction_memory(
        agent_name=self._agent_name,
        cooperation_scores=cooperation_scores,
        trustee_circle=self._trustee_circle,
    )


### ~~~~ QUESTIONS ~~~~ ###


class BestOptionPerception(
    question_of_recent_memories.QuestionOfRecentMemories
):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "Given the statements above, which of {agent_name}'s options has"
            ' the highest likelihood of causing {agent_name} to achieve their'
            ' goal? If multiple options have the same likelihood, select in'
            ' priority the one that follows the strategy the most.'
        ),
        answer_prefix="{agent_name}'s best course of action is ",
        add_to_memory=False,
        **kwargs,
    )


### ~~~~ AGENT BUILD ~~~~ ###


def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
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
    raise ValueError(
        'This function is meant for a main character '
        'but it was called on a supporting character.'
    )
  agent_name = config.name
  trustee_circle = TrusteeCircle()

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

  observation_label = 'Observation'
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
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel('ObservationSummary').on_next,
  )

  relevant_memories_label = 'Recalled memories and observations'
  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      model=model,
      components={
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(time_display): 'The current date/time is',
      },
      num_memories_to_retrieve=10,
      pre_act_key=relevant_memories_label,
      logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
  )

  trustee_circle_label = 'Strategy'
  trustee_circle = ObservationTrust(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=trustee_circle_label,
      logging_channel=measurements.get_channel('TrusteeCircle').on_next,
  )

  if config.goal:
    goal_label = 'Overarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next,
    )
  else:
    goal_label = None
    overarching_goal = None

  options_perception_label = (
      f'\nQuestion: Which options are available to {agent_name} '
      'right now?\nAnswer'
  )
  options_perception = (
      agent_components.question_of_recent_memories.AvailableOptionsPerception(
          model=model,
          components={
              _get_class_name(observation): observation_label,
              _get_class_name(observation_summary): observation_summary_label,
              _get_class_name(relevant_memories): relevant_memories_label,
              goal_label: goal_label,
          },
          clock_now=clock.now,
          pre_act_key=options_perception_label,
          logging_channel=measurements.get_channel(
              'AvailableOptionsPerception'
          ).on_next,
      )
  )

  best_option_perception_label = (
      f'\nQuestion: Of the options available to {agent_name}, and '
      'given their goal, which choice of action or strategy is '
      f'best for {agent_name} to take right now?\nAnswer'
  )
  best_option_perception = BestOptionPerception(
      model=model,
      components={
          _get_class_name(observation): observation_label,
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(relevant_memories): relevant_memories_label,
          _get_class_name(trustee_circle): trustee_circle_label,
          _get_class_name(options_perception): options_perception_label,
          goal_label: goal_label,
      },
      clock_now=clock.now,
      pre_act_key=best_option_perception_label,
      logging_channel=measurements.get_channel('BestOptionPerception').on_next,
  )

  entity_components = (
      instructions,
      time_display,
      observation,
      observation_summary,
      relevant_memories,
      options_perception,
      trustee_circle,
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
    components_of_agent[goal_label] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, goal_label)

  print(component_order)

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
  )

  return agent
