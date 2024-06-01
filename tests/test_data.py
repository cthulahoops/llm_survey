import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, from_dtype
from llm_survey.data import ModelOutput, SurveyDb


@given(
    output=st.builds(
        ModelOutput,
        id=st.none(),
        model=st.text(),
        content=st.text(),
        embedding=arrays(
            np.float64,
            shape=(10,),
            elements=from_dtype(
                np.dtype("float64"), allow_nan=False, allow_infinity=False
            ),
        ),
    )
)
def test_save_and_retrieve_output(output):
    db = SurveyDb(":memory:")
    db.create_tables()

    output_id = db.save_model_output(output)
    assert isinstance(output_id, int)
    output.id = output_id
    actual_output = db.get_model_output(output_id)

    assert all(output.embedding == actual_output.embedding)
