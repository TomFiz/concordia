# @title Imports for agent building
import datetime
import json
import numpy as np  # For calculating average and std deviation
import statistics
from typing import List, Dict, Sequence, Tuple, Union, Callable

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.components.agent import memory_component
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.typing import entity_component
from concordia.components.agent import action_spec_ignored
from concordia.typing import entity as entity_lib
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
    print(
        f"Agent '{ext_agent_name}' updated with cooperation score"
        f' {cooperation_score}.'
    )

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


DEFAULT_OBSERVATION_TRUSTEES_PRE_ACT_KEY = 'Circle of Trustees'


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
          f'{self._agent_name} cannot trust anyone at the moment, and should'
          ' act for his own interest.'
      )

    else:
      return (
          f'{self._agent_name} can trust the following persons : '
          f' {trusted_agents}.{self._agent_name} must be willing to cooperate'
          ' with them to reach a bigger collective objective.'
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
          f" {{agent_name: int}}.Here is the interaction : '{observations}'."
      )

      response = model.sample_text(prompt=prompt, temperature=0.0)

      try:
        res_dict = json.loads(
            response
        )  # Convert the model response to a dictionary
        if isinstance(res_dict, dict):
          return res_dict  # Return a dictionary of agent names and their respective cooperation scores
        else:
          return {}
      except (json.JSONDecodeError, TypeError):
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


class QuestionIdentity(question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      agent_name: str,
      entity: entity_component.EntityWithComponents = None,
      **kwargs,
  ):
    # @markdown {agent_name} will be automatically replaced with the name of the specific agent
    question = (  # @param {"type":"string"}
        'Given the above, what kind of person is {agent_name}?'
    )
    # @markdown The answer will have to start with this prefix
    answer_prefix = '{agent_name} is '  # @param {"type":"string"}
    # @markdown Flag that defines whether the answer will be added to memory
    add_to_memory = True  # @param {"type":"boolean"}
    # @markdown If yes, the memory will start with this tag
    memory_tag = '[self reflection]'  # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    self._entity = entity
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={},
        **kwargs,
    )


# @markdown We can add the value of other components to the context of the question. Notice, how QuestionSituation depends on Observation and ObservationSummary. The names of the classes of the contextualising components have to be passed as "components" argument.
class QuestionSituation(question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers 'which situation is the agent in at this moment ?'."""

  def __init__(
      self,
      agent_name: str,
      entity: entity_component.EntityWithComponents = None,
      **kwargs,
  ):
    question = (  # @param {"type":"string"}
        'Given the statements above, what kind of situation is {agent_name} in'
        ' right now?'
    )
    answer_prefix = '{agent_name} is currently '  # @param {"type":"string"}
    add_to_memory = False  # @param {"type":"boolean"}
    memory_tag = '[situation reflection]'  # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    self._entity = entity

    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        # @markdown The key is the name of the component class and the key is the prefix with which it will appear in the context of this component. Be careful if you are going to edit this field, it should be a valid dictionary.
        components={
            'Observation': '\nObservation',
            'ObservationSummary': '\nSummary of recent observations',
        },  # @param
        **kwargs,
    )


class QuestionAction(question_of_recent_memories.QuestionOfRecentMemories):
  """What would a person like the agent do in a situation like this?"""

  def __init__(
      self,
      agent_name: str,
      entity: entity_component.EntityWithComponents = None,
      **kwargs,
  ):
    question = (  # @param {"type":"string"}
        'Knowing the above, what would a person like {agent_name} do in a'
        ' situation like this ?'
    )
    answer_prefix = '{agent_name} would '  # @param {"type":"string"}
    add_to_memory = True  # @param {"type":"boolean"}
    memory_tag = '[intent reflection]'  # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    self._entity = entity

    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={
            'QuestionIdentity': (
                f'\nQuestion: What kind of person is {agent_name}?\nAnswer'
            ),
            'QuestionSituation': (
                f'\nQuestion: What kind of situation is {agent_name} in right'
                ' now?\nAnswer'
            ),
            'ObservationTrust': f'\n{DEFAULT_OBSERVATION_TRUSTEES_PRE_ACT_KEY}',
        },  # @param
        **kwargs,
    )


def _make_trustee_question_components(
    agent_name: str,
    measurements: measurements_lib.Measurements,
    model: language_model.LanguageModel,
    clock: game_clock.MultiIntervalClock,
) -> Sequence[question_of_recent_memories.QuestionOfRecentMemories]:

  question_1 = QuestionIdentity(
      agent_name=agent_name,
      model=model,
      logging_channel=measurements.get_channel('Question_1').on_next,
  )
  question_2 = QuestionSituation(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_2').on_next,
  )
  question_3 = QuestionAction(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_3').on_next,
  )

  return (question_1, question_2, question_3)


class CustomConcatActComponent(
    agent_components.concat_act_component.ConcatActComponent
):
  """Custom act component that adds a custom question to the context."""

  def __init__(self, *args, trustee_circle, agent_name, model, **kwargs):
    super().__init__(model=model, *args, **kwargs)
    self.trustee_circle = trustee_circle
    self.agent_name = agent_name
    self.model = model

  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:

    # Answer Question 1
    answer_1 = contexts['QuestionIdentity']
    # Answer Question 2
    answer_2 = contexts['QuestionSituation']

    # Answer QuestionAction
    answer_action_trust = contexts['QuestionAction']

    return super().get_action_attempt(contexts, action_spec)


def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> entity_agent_with_logging.EntityAgentWithLogging:

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

  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key='\nObservation',
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

  relevant_memories_label = '\nRecalled memories and observations'
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

  trustee_circle_label = 'Trustee Circle'
  trustee_circle = ObservationTrust(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=trustee_circle_label,
      logging_channel=measurements.get_channel('TrusteeCircle').on_next,
  )

  question_components = _make_trustee_question_components(
      agent_name=agent_name,
      model=model,
      clock=clock,
      measurements=measurements,
  )

  core_components = (
      instructions,
      time_display,
      observation,
      observation_summary,
      relevant_memories,
      trustee_circle,
  )
  entity_components = core_components + tuple(question_components)
  components_of_agent = {
      _get_class_name(component): component for component in entity_components
  }
  components_of_agent[
      agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME
  ] = agent_components.memory_component.MemoryComponent(raw_memory)
  component_order = list(components_of_agent.keys())

  act_component = CustomConcatActComponent(
      agent_name=agent_name,
      model=model,
      clock=clock,
      component_order=component_order,
      logging_channel=measurements.get_channel('ActComponent').on_next,
      trustee_circle=trustee_circle,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
  )

  return agent
