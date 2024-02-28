from gluonts.dataset.repository.datasets import get_dataset
from gluonts.transform import AddAgeFeature, Chain, TransformedDataset
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.evaluation import make_evaluation_predictions
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Get dataset
dataset = get_dataset("exchange_rate", regenerate=True)
print(dataset.metadata)

# Train and predict with AddAgeFeature
transformation_with_age = Chain(
    [
        AddAgeFeature(
            output_field="age",
            target_field="target",
            pred_length=dataset.metadata.prediction_length,
        )
    ]
)

transformed_train_with_age = TransformedDataset(dataset.train, transformation_with_age)

estimator_with_age = SimpleFeedForwardEstimator(
    hidden_dimensions=[10],
    prediction_length=dataset.metadata.prediction_length,
    context_length=100,
    trainer_kwargs={"max_epochs": 100},
)

predictor_with_age = estimator_with_age.train(transformed_train_with_age)

forecast_it_with_age, ts_it_with_age = make_evaluation_predictions(
    dataset=dataset.test,
    predictor=predictor_with_age,
    num_samples=100,
)
forecasts_with_age = list(forecast_it_with_age)
tss_with_age = list(ts_it_with_age)

# Train and predict without AddAgeFeature
estimator_without_age = SimpleFeedForwardEstimator(
    hidden_dimensions=[10],
    prediction_length=dataset.metadata.prediction_length,
    context_length=100,
    trainer_kwargs={"max_epochs": 100},
)

predictor_without_age = estimator_without_age.train(dataset.train)

forecast_it_without_age, ts_it_without_age = make_evaluation_predictions(
    dataset=dataset.test,
    predictor=predictor_without_age,
    num_samples=100,
)
forecasts_without_age = list(forecast_it_without_age)
tss_without_age = list(ts_it_without_age)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot with AddAgeFeature
ts_entry_with_age = tss_with_age[0]
ax[0].plot(ts_entry_with_age[-150:].to_timestamp())
forecasts_with_age[0].plot(show_label=True, ax=ax[0], intervals=())
ax[0].set_title("Forecast with AddAgeFeature")
ax[0].legend()

# Plot without AddAgeFeature
ts_entry_without_age = tss_without_age[0]
ax[1].plot(ts_entry_without_age[-150:].to_timestamp())
forecasts_without_age[0].plot(show_label=True, ax=ax[1], intervals=())
ax[1].set_title("Forecast without AddAgeFeature")
ax[1].legend()

plt.tight_layout()
plt.show()
