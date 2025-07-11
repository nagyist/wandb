package runbranch_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wandb/wandb/core/internal/filestream"
	"github.com/wandb/wandb/core/internal/gqlmock"
	"github.com/wandb/wandb/core/internal/runbranch"
	"github.com/wandb/wandb/core/internal/runconfig"
)

type ResumeResponse struct {
	Model Model `json:"model"`
}
type Model struct {
	Bucket Bucket `json:"bucket"`
}
type Bucket struct {
	Name             string   `json:"name"`
	HistoryLineCount *int     `json:"historyLineCount"`
	EventsLineCount  *int     `json:"eventsLineCount"`
	LogLineCount     *int     `json:"logLineCount"`
	HistoryTail      *string  `json:"historyTail"`
	SummaryMetrics   *string  `json:"summaryMetrics"`
	Config           *string  `json:"config"`
	EventsTail       string   `json:"eventsTail"`
	Tags             []string `json:"tags"`
	WandbConfig      string   `json:"wandbConfig"`
	Id               string   `json:"id"`
	Notes            *string  `json:"notes"`
}

func TestNeverResumeEmptyResponse(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()
	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		`{}`,
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"never")
	err := resumeState.UpdateForResume(&runbranch.RunParams{}, runconfig.New())
	assert.Nil(t, err, "GetUpdates should not return an error")
}

func TestAllowResumeEmptyResponse(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()
	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		`{}`,
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"allow")
	err := resumeState.UpdateForResume(&runbranch.RunParams{}, runconfig.New())
	assert.Nil(t, err, "GetUpdates should not return an error")
}

func TestMustResumeEmptyResponse(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()
	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		`{}`,
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")
	err := resumeState.UpdateForResume(&runbranch.RunParams{}, runconfig.New())
	assert.NotNil(t, err, "GetUpdates should return an error")
	assert.IsType(t, &runbranch.BranchError{}, err, "GetUpdates should return a BranchError")
	assert.NotNil(t, err.(*runbranch.BranchError).Response, "BranchError should have a response")
}

func TestMustResumeNilResponse(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()
	nilResponse, _ := json.Marshal(nil)
	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(nilResponse),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")
	err := resumeState.UpdateForResume(&runbranch.RunParams{}, runconfig.New())
	assert.NotNil(t, err, "GetUpdates should return an error")
	assert.IsType(t, &runbranch.BranchError{}, err, "GetUpdates should return a BranchError")
	assert.NotNil(t, err.(*runbranch.BranchError).Response, "BranchError should have a response")
}

func TestNeverResumeNoneEmptyResponse(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()
	history := "[]"
	config := "{}"
	summary := "{}"
	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:           "FakeName",
				HistoryTail:    &history,
				SummaryMetrics: &summary,
				Config:         &config,
				EventsTail:     "[]",
				WandbConfig:    `{"t": 1}`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")
	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"never")
	err = resumeState.UpdateForResume(&runbranch.RunParams{}, runconfig.New())
	assert.NotNil(t, err, "GetUpdates should return an error")
	assert.IsType(t, &runbranch.BranchError{}, err, "GetUpdates should return a BranchError")
	assert.NotNil(t, err.(*runbranch.BranchError).Response, "BranchError should have a response")
}

func TestMustResumeNoTelemetryInConfig(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()
	history := "[]"
	config := "{}"
	summary := "{}"
	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:           "FakeName",
				HistoryTail:    &history,
				SummaryMetrics: &summary,
				Config:         &config,
				EventsTail:     "[]",
				WandbConfig:    `{}`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")
	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")
	err = resumeState.UpdateForResume(&runbranch.RunParams{}, runconfig.New())
	assert.NotNil(t, err, "GetUpdates should return an error")
	assert.IsType(t, &runbranch.BranchError{}, err, "GetUpdates should return a BranchError")
}

func TestAllowResumeNoneEmptyResponse(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()

	historyLineCount := 0
	eventsLineCount := 0
	logLineCount := 0
	history := "[]"
	config := "{}"
	summary := "{}"
	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "FakeName",
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				HistoryTail:      &history,
				SummaryMetrics:   &summary,
				Config:           &config,
				EventsTail:       "[]",
				WandbConfig:      `{"t": 1}`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")

	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"allow")
	err = resumeState.UpdateForResume(&runbranch.RunParams{}, runconfig.New())
	assert.Nil(t, err, "GetUpdates should not return an error")
}

func TestMustResumeNoneEmptyResponse(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()

	historyLineCount := 0
	eventsLineCount := 0
	logLineCount := 0
	hisotry := "[]"
	config := "{}"
	summary := "{}"
	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "FakeName",
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				HistoryTail:      &hisotry,
				SummaryMetrics:   &summary,
				Config:           &config,
				EventsTail:       "[]",
				WandbConfig:      `{"t": 1}`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")
	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")
	err = resumeState.UpdateForResume(&runbranch.RunParams{}, runconfig.New())
	assert.Nil(t, err, "GetUpdates should not return an error")
}

func TestMustResumeValidHistory(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()

	history := `["{\"_step\":1,\"_runtime\":50}"]`
	config := "{}"
	summary := `{"_step": 1, "_runtime": 50}`
	historyLineCount := 1
	eventsLineCount := 0
	logLineCount := 0
	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "FakeName",
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				HistoryTail:      &history,
				SummaryMetrics:   &summary,
				Config:           &config,
				EventsTail:       "[]",
				WandbConfig:      `{"t": 1}`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")

	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")
	params := &runbranch.RunParams{}
	err = resumeState.UpdateForResume(params, runconfig.New())
	assert.Equal(t, int64(2), params.StartingStep, "GetUpdates should return correct starting step")
	assert.Equal(t, int32(50), params.Runtime, "GetUpdates should return correct runtime")
	assert.True(t, params.Resumed, "GetUpdates should return correct resumed state")
	assert.Nil(t, err, "GetUpdates should not return an error")
}

func TestMustResumeZeroHisotry(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()

	historyLineCount := 0
	eventsLineCount := 0
	logLineCount := 0
	history := "[]"
	config := "{}"
	summary := "{}"
	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "FakeName",
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				HistoryTail:      &history,
				SummaryMetrics:   &summary,
				Config:           &config,
				EventsTail:       "[]",
				WandbConfig:      `{"t": 1}`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")

	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")
	params := &runbranch.RunParams{}
	err = resumeState.UpdateForResume(params, runconfig.New())
	assert.Equal(t, int64(0), params.StartingStep, "GetUpdates should return correct starting step")
	assert.Equal(t, int32(0), params.Runtime, "GetUpdates should return correct runtime")
	assert.True(t, params.Resumed, "GetUpdates should return correct resumed state")
	assert.Nil(t, err, "GetUpdates should not return an error")
}

func TestMustResumeHistoryTailStepZero(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()

	historyLineCount := 0
	eventsLineCount := 0
	logLineCount := 0
	history := `["{\"_step\":1}"]`
	config := "{}"
	summary := `{"_step": 1}`
	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "FakeName",
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				HistoryTail:      &history,
				SummaryMetrics:   &summary,
				Config:           &config,
				EventsTail:       "[]",
				WandbConfig:      `{"t": 1}`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")

	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")

	params := &runbranch.RunParams{}
	err = resumeState.UpdateForResume(params, runconfig.New())
	assert.Equal(t, int64(1), params.StartingStep, "GetUpdates should return correct starting step")
	assert.Equal(t, int32(0), params.Runtime, "GetUpdates should return correct runtime")
	assert.True(t, params.Resumed, "GetUpdates should return correct resumed state")
	assert.Nil(t, err, "GetUpdates should not return an error")
}

func TestMustResumeValidSummary(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()

	history := `["{\"_step\":1, \"_runtime\":40.2}"]`
	config := "{}"
	summary := `{"loss": 0.5, "_runtime": 40.2, "_wandb": {"runtime": 20.3}, "_step": 1}`
	historyLineCount := 1
	eventsLineCount := 0
	logLineCount := 0
	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "FakeName",
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				HistoryTail:      &history,
				SummaryMetrics:   &summary,
				Config:           &config,
				EventsTail:       "[]",
				WandbConfig:      `{"t": 1}`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")

	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")

	params := &runbranch.RunParams{}
	err = resumeState.UpdateForResume(params, runconfig.New())
	assert.Equal(t, int64(2), params.StartingStep, "GetUpdates should return correct starting step")
	assert.Equal(t, int32(40), params.Runtime, "GetUpdates should return correct runtime")
	assert.True(t, params.Resumed, "GetUpdates should return correct resumed state")

	// check the value of the summary are correct
	assert.Len(t, params.Summary, 4, "GetUpdates should return correct summary")
	assert.Equal(t, 0.5, params.Summary["loss"], "GetUpdates should return correct summary")
	assert.Equal(t, float64(40.2), params.Summary["_runtime"], "GetUpdates should return correct summary")
	assert.Equal(t, float64(20.3), params.Summary["_wandb"].(map[string]any)["runtime"], "GetUpdates should return correct summary")

	assert.Nil(t, err, "GetUpdates should not return an error")
}

func TestMustResumeValidConfig(t *testing.T) {

	mockGQL := gqlmock.NewMockClient()

	historyLineCount := 0
	eventsLineCount := 0
	logLineCount := 0
	history := "[]"
	configStr := `{"lr": {"value": 0.001}}`
	summary := "{}"
	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "FakeName",
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				HistoryTail:      &history,
				SummaryMetrics:   &summary,
				Config:           &configStr,
				EventsTail:       "[]",
				WandbConfig:      `{"t": 1}`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")

	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")

	params := &runbranch.RunParams{}
	config := runconfig.New()
	err = resumeState.UpdateForResume(params, config)
	assert.Nil(t, err, "GetUpdates should not return an error")
	assert.Equal(t, int64(0), params.StartingStep, "GetUpdates should return correct starting step")
	assert.Equal(t, int32(0), params.Runtime, "GetUpdates should return correct runtime")
	assert.True(t, params.Resumed, "GetUpdates should return correct resumed state")
	assert.Len(t, config.CloneTree(), 1, "GetUpdates should return correct config")
	assert.Equal(t, 0.001, config.CloneTree()["lr"], "GetUpdates should return correct config")
}

func TestMustResumeValidTags(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()

	historyLineCount := 0
	eventsLineCount := 0
	logLineCount := 0
	history := "[]"
	config := "{}"
	summary := "{}"
	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "FakeName",
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				HistoryTail:      &history,
				SummaryMetrics:   &summary,
				Config:           &config,
				EventsTail:       "[]",
				Tags:             []string{"tag1", "tag2"},
				WandbConfig:      `{"t": 1}`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")

	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")

	params := &runbranch.RunParams{}
	err = resumeState.UpdateForResume(params, runconfig.New())
	assert.Nil(t, err, "GetUpdates should not return an error")
	assert.Equal(t, int64(0), params.StartingStep, "GetUpdates should return correct starting step")
	assert.Equal(t, int32(0), params.Runtime, "GetUpdates should return correct runtime")
	assert.True(t, params.Resumed, "GetUpdates should return correct resumed state")
	assert.Len(t, params.Tags, 2, "GetUpdates should return correct tags")
	assert.Contains(t, params.Tags, "tag1", "GetUpdates should return correct tags")
	assert.Contains(t, params.Tags, "tag2", "GetUpdates should return correct tags")
}

func TestMustResumeValidStorageId(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()

	historyLineCount := 0
	eventsLineCount := 0
	logLineCount := 0
	history := "[]"
	config := "{}"
	summary := "{}"
	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "FakeName",
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				HistoryTail:      &history,
				SummaryMetrics:   &summary,
				Config:           &config,
				EventsTail:       "[]",
				WandbConfig:      `{"t": 1}`,
				Id:               `storage_id`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")

	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")

	params := &runbranch.RunParams{}
	err = resumeState.UpdateForResume(params, runconfig.New())
	assert.Nil(t, err, "GetUpdates should not return an error")
	assert.Equal(t, int64(0), params.StartingStep, "GetUpdates should return correct starting step")
	assert.Equal(t, int32(0), params.Runtime, "GetUpdates should return correct runtime")
	assert.True(t, params.Resumed, "GetUpdates should return correct resumed state")
	assert.Equal(t, "storage_id", params.StorageID, "GetUpdates should return correct storage id")
}

func TestMustResumeValidEvents(t *testing.T) {

	mockGQL := gqlmock.NewMockClient()

	historyLineCount := 0
	eventsLineCount := 0
	logLineCount := 0
	history := `["{\"_runtime\":10}"]`
	config := "{}"
	summary := `{ "_runtime": 20 }`
	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "FakeName",
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				HistoryTail:      &history,
				SummaryMetrics:   &summary,
				Config:           &config,
				EventsTail:       `["{\"_runtime\":40}", "{\"_runtime\":50}"]`,
				Tags:             []string{"tag1", "tag2"},
				WandbConfig:      `{"t": 1}`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")

	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")

	params := &runbranch.RunParams{}
	err = resumeState.UpdateForResume(params, runconfig.New())
	assert.Nil(t, err, "GetUpdates should not return an error")
	assert.Equal(t, int64(0), params.StartingStep, "GetUpdates should return correct starting step")
	assert.Equal(t, int32(50), params.Runtime, "GetUpdates should return correct runtime")
	assert.True(t, params.Resumed, "GetUpdates should return correct resumed state")

	assert.Len(t, params.Summary, 1, "GetUpdates should return correct summary")
	assert.Equal(t, int64(20), params.Summary["_runtime"], "GetUpdates should return correct summary")
}

func TestMustResumeNullValue(t *testing.T) {

	historyLineCount := 0
	eventsLineCount := 0
	logLineCount := 0
	config := "{}"
	summary := "{}"
	history := "[]"
	testCase := []struct {
		name     string
		response ResumeResponse
	}{
		{
			name: "NullHistory",
			response: ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:             "FakeName",
						HistoryLineCount: &historyLineCount,
						EventsLineCount:  &eventsLineCount,
						LogLineCount:     &logLineCount,
						SummaryMetrics:   &summary,
						Config:           &config,
						EventsTail:       "[]",
						WandbConfig:      `{"t": 1}`,
					},
				},
			},
		},
		{
			name: "NullSummary",
			response: ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:             "FakeName",
						HistoryLineCount: &historyLineCount,
						EventsLineCount:  &eventsLineCount,
						LogLineCount:     &logLineCount,
						HistoryTail:      &history,
						Config:           &config,
						EventsTail:       "[]",
					},
				},
			},
		},
		{
			name: "NullConfig",
			response: ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:             "FakeName",
						HistoryLineCount: &historyLineCount,
						EventsLineCount:  &eventsLineCount,
						LogLineCount:     &logLineCount,
						HistoryTail:      &history,
						SummaryMetrics:   &summary,
						EventsTail:       "[]",
					},
				},
			},
		},
	}
	for _, tc := range testCase {
		t.Run(tc.name, func(t *testing.T) {

			mockGQL := gqlmock.NewMockClient()

			jsonData, err := json.MarshalIndent(tc.response, "", "    ")
			assert.Nil(t, err, "Failed to marshal json data")

			mockGQL.StubMatchOnce(
				gqlmock.WithOpName("RunResumeStatus"),
				string(jsonData),
			)
			resumeState := runbranch.NewResumeBranch(
				context.Background(),
				mockGQL,
				"must")

			err = resumeState.UpdateForResume(&runbranch.RunParams{}, runconfig.New())
			assert.NotNil(t, err, "GetUpdates should return an error")
			assert.IsType(t, &runbranch.BranchError{}, err, "GetUpdates should return a BranchError")
			assert.NotNil(t, err.(*runbranch.BranchError).Response, "BranchError should have a response")
		})
	}
}

func TestAllowResumeNullValue(t *testing.T) {
	config := "{}"
	summary := "{}"
	history := "[]"
	testCase := []struct {
		name     string
		response ResumeResponse
	}{
		{
			name: "NullHistory",
			response: ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:           "FakeName",
						SummaryMetrics: &summary,
						Config:         &config,
						EventsTail:     "[]",
						WandbConfig:    `{"t": 1}`,
					},
				},
			},
		},
		{
			name: "NullSummary",
			response: ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:        "FakeName",
						HistoryTail: &history,
						Config:      &config,
						EventsTail:  "[]",
						WandbConfig: `{"t": 1}`,
					},
				},
			},
		},
		{
			name: "NullConfig",
			response: ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:           "FakeName",
						HistoryTail:    &history,
						SummaryMetrics: &summary,
						EventsTail:     "[]",
						WandbConfig:    `{"t": 1}`,
					},
				},
			},
		},
	}
	for _, tc := range testCase {
		t.Run(tc.name, func(t *testing.T) {

			mockGQL := gqlmock.NewMockClient()

			jsonData, err := json.MarshalIndent(tc.response, "", "    ")
			assert.Nil(t, err, "Failed to marshal json data")

			mockGQL.StubMatchOnce(
				gqlmock.WithOpName("RunResumeStatus"),
				string(jsonData),
			)
			resumeState := runbranch.NewResumeBranch(
				context.Background(),
				mockGQL,
				"allow")

			err = resumeState.UpdateForResume(&runbranch.RunParams{}, runconfig.New())
			assert.NotNil(t, err, "GetUpdates should return an error")
			if _, ok := err.(*runbranch.BranchError); ok {
				t.Errorf("expected a BranchError but got %T", err)
			}
		})
	}
}

func TestMustResumeInvalidHistory(t *testing.T) {

	testCases := []struct {
		name  string
		value string
	}{
		{
			name:  "InvalidContent",
			value: `["invalid_history"]`,
		},
		{
			name:  "InvalidShape",
			value: `{"_step":0}`,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockGQL := gqlmock.NewMockClient()

			config := "{}"
			summary := "{}"
			historyLineCount := 0
			eventsLineCount := 0
			logLineCount := 0
			rr := ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:             "FakeName",
						HistoryLineCount: &historyLineCount,
						EventsLineCount:  &eventsLineCount,
						LogLineCount:     &logLineCount,
						HistoryTail:      &tc.value,
						SummaryMetrics:   &summary,
						Config:           &config,
						EventsTail:       `[]`,
						WandbConfig:      `{"t": 1}`,
					},
				},
			}

			jsonData, err := json.MarshalIndent(rr, "", "    ")
			assert.Nil(t, err, "Failed to marshal json data")

			mockGQL.StubMatchOnce(
				gqlmock.WithOpName("RunResumeStatus"),
				string(jsonData),
			)
			resumeState := runbranch.NewResumeBranch(
				context.Background(),
				mockGQL,
				"must")

			err = resumeState.UpdateForResume(&runbranch.RunParams{}, runconfig.New())
			assert.NotNil(t, err, "GetUpdates should return an error")
			assert.IsType(t, &runbranch.BranchError{}, err, "GetUpdates should return a BranchError")
			assert.NotNil(t, err.(*runbranch.BranchError).Response, "BranchError should have a response")
		})
	}
}

func TestMustResumeInvalidSummary(t *testing.T) {

	mockGQL := gqlmock.NewMockClient()

	history := `[]`
	config := "{}"
	summary := `[]`
	historyLineCount := 0
	eventsLineCount := 0
	logLineCount := 0
	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "FakeName",
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				HistoryTail:      &history,
				SummaryMetrics:   &summary,
				Config:           &config,
				EventsTail:       `[]`,
				WandbConfig:      `{"t": 1}`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")

	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)
	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")

	err = resumeState.UpdateForResume(&runbranch.RunParams{}, runconfig.New())
	assert.NotNil(t, err, "GetUpdates should return an error")
	assert.IsType(t, &runbranch.BranchError{}, err, "GetUpdates should return a BranchError")
	assert.NotNil(t, err.(*runbranch.BranchError).Response, "BranchError should have a response")
	assert.Nil(t, nil, "GetUpdates should return nil when response is empty")
}

func TestMustResumeInvalidConfig(t *testing.T) {

	testCases := []struct {
		name  string
		value string
	}{
		{
			name:  "ConfigList",
			value: `[]`,
		},
		{
			name:  "ConfigNotNested",
			value: `{"_step":0}`,
		},
		{
			name:  "ConfigNestedNotValue",
			value: `{"_step": {"runtime": 30}`,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			history := `[]`
			summary := `{}`
			mockGQL := gqlmock.NewMockClient()
			historyLineCount := 0
			eventsLineCount := 0
			logLineCount := 0
			rr := ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:             "FakeName",
						HistoryLineCount: &historyLineCount,
						EventsLineCount:  &eventsLineCount,
						LogLineCount:     &logLineCount,
						HistoryTail:      &history,
						SummaryMetrics:   &summary,
						Config:           &tc.value,
						EventsTail:       `[]`,
						WandbConfig:      `{"t": 1}`,
					},
				},
			}

			jsonData, err := json.MarshalIndent(rr, "", "    ")
			assert.Nil(t, err, "Failed to marshal json data")

			mockGQL.StubMatchOnce(
				gqlmock.WithOpName("RunResumeStatus"),
				string(jsonData),
			)
			resumeState := runbranch.NewResumeBranch(
				context.Background(),
				mockGQL,
				"must")

			err = resumeState.UpdateForResume(&runbranch.RunParams{}, runconfig.New())
			assert.NotNil(t, err, "GetUpdates should return an error")
			assert.IsType(t, &runbranch.BranchError{}, err, "GetUpdates should return a BranchError")
			assert.NotNil(t, err.(*runbranch.BranchError).Response, "BranchError should have a response")
		})
	}
}

func TestNotNeverResumeFileStreamOffset(t *testing.T) {

	history := `[]`
	summary := `{}`
	config := `{}`

	testCases := []struct {
		name  string
		value string
	}{
		{
			name:  "Allow",
			value: "allow",
		},
		{
			name:  "Must",
			value: "must",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockGQL := gqlmock.NewMockClient()
			historyLineCount := 10
			eventsLineCount := 13
			logLineCount := 15
			rr := ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:             "FakeName",
						HistoryLineCount: &historyLineCount,
						EventsLineCount:  &eventsLineCount,
						LogLineCount:     &logLineCount,
						HistoryTail:      &history,
						SummaryMetrics:   &summary,
						Config:           &config,
						EventsTail:       `[]`,
						WandbConfig:      `{"t": 1}`,
					},
				},
			}

			jsonData, err := json.MarshalIndent(rr, "", "    ")
			assert.Nil(t, err, "Failed to marshal json data")

			mockGQL.StubMatchOnce(
				gqlmock.WithOpName("RunResumeStatus"),
				string(jsonData),
			)
			resumeState := runbranch.NewResumeBranch(
				context.Background(),
				mockGQL,
				tc.value)
			params := &runbranch.RunParams{}
			err = resumeState.UpdateForResume(params, runconfig.New())
			assert.Nil(t, err, "GetUpdates should not return an error")
			assert.Len(t, params.FileStreamOffset, 3, "GetUpdates should return correct file stream offset")
			assert.Equal(t, 10, params.FileStreamOffset[filestream.HistoryChunk], "GetUpdates should return correct file stream offset")
			assert.Equal(t, 13, params.FileStreamOffset[filestream.EventsChunk], "GetUpdates should return correct file stream offset")
			assert.Equal(t, 15, params.FileStreamOffset[filestream.OutputChunk], "GetUpdates should return correct file stream offset")
		})
	}
}

func TestExtractRunState(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()

	historyLineCount := 5
	eventsLineCount := 10
	logLineCount := 15
	history := `["{\"_step\":4,\"_runtime\":100}"]`
	summary := `{"loss": 0.5, "_runtime": 120, "_wandb": {"runtime": 130}, "_step": 4}`
	configStr := `{"lr": {"value": 0.001}, "batch_size": {"value": 32}}`
	eventsTail := `["{\"_runtime\":110}", "{\"_runtime\":120}"]`
	storageId := "test_storage_id"

	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "TestRun",
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				HistoryTail:      &history,
				SummaryMetrics:   &summary,
				Config:           &configStr,
				EventsTail:       eventsTail,
				Tags:             []string{"test", "extract"},
				WandbConfig:      `{"t": 1}`,
				Id:               storageId,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")

	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)

	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"allow")

	params := &runbranch.RunParams{
		Entity:  "test-entity",
		Project: "test-project",
		RunID:   "test-run-id",
	}
	config := runconfig.New()
	err = resumeState.UpdateForResume(params, config)

	assert.Nil(t, err, "GetUpdates should not return an error")

	// Test FileStreamOffset
	assert.Equal(t, historyLineCount, params.FileStreamOffset[filestream.HistoryChunk], "Incorrect history line count")
	assert.Equal(t, eventsLineCount, params.FileStreamOffset[filestream.EventsChunk], "Incorrect events line count")
	assert.Equal(t, logLineCount, params.FileStreamOffset[filestream.OutputChunk], "Incorrect log line count")

	// Test StartingStep
	assert.Equal(t, int64(5), params.StartingStep, "Incorrect starting step")

	// Test Runtime
	assert.Equal(t, int32(130), params.Runtime, "Incorrect runtime")

	// Test Resumed flag
	assert.True(t, params.Resumed, "Resumed flag should be true")

	// Test Summary
	assert.Equal(t, 0.5, params.Summary["loss"], "Incorrect loss in summary")
	assert.Equal(t, int64(120), params.Summary["_runtime"], "Incorrect _runtime in summary")
	wandbRuntime, ok := params.Summary["_wandb"].(map[string]interface{})["runtime"]
	assert.True(t, ok, "wandb runtime not found in summary")
	assert.Equal(t, int64(130), wandbRuntime, "Incorrect wandb runtime in summary")

	// Test Config
	configTree := config.CloneTree()
	assert.Equal(t, 0.001, configTree["lr"], "Incorrect learning rate in config")
	assert.Equal(t, int64(32), configTree["batch_size"], "Incorrect batch size in config")

	// Test Tags
	assert.Equal(t, []string{"test", "extract"}, params.Tags, "Incorrect tags")

	// Test GQL ID
	assert.Equal(t, storageId, params.StorageID, "GetUpdates should set storage id")
}

func TestExtractRunStateNilCases(t *testing.T) {
	historyLineCount := 5
	eventsLineCount := 10
	logLineCount := 15
	history := `["{\"_step\":4,\"_runtime\":100}"]`
	summary := `{"loss": 0.5, "_runtime": 120, "_wandb": {"runtime": 130}}`
	config := `{"lr": {"value": 0.001}, "batch_size": {"value": 32}}`
	testCases := []struct {
		name          string
		response      ResumeResponse
		expectError   bool
		errorContains string
	}{
		{
			name: "Nil HistoryLineCount",
			response: ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:            "TestRun",
						EventsLineCount: &eventsLineCount,
						LogLineCount:    &logLineCount,
						HistoryTail:     &history,
						SummaryMetrics:  &summary,
						EventsTail:      "[]",
						WandbConfig:     `{"t": 1}`,
						Config:          &config,
					},
				},
			},
			expectError:   true,
			errorContains: "no history line count found",
		},
		{
			name: "Nil EventsLineCount",
			response: ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:             "TestRun",
						HistoryLineCount: &historyLineCount,
						LogLineCount:     &logLineCount,
						HistoryTail:      &history,
						SummaryMetrics:   &summary,
						EventsTail:       "[]",
						Config:           &config,
						WandbConfig:      `{"t": 1}`,
					},
				},
			},
			expectError:   true,
			errorContains: "no events line count found",
		},
		{
			name: "Nil LogLineCount",
			response: ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:             "TestRun",
						HistoryLineCount: &historyLineCount,
						EventsLineCount:  &eventsLineCount,
						HistoryTail:      &history,
						SummaryMetrics:   &summary,
						EventsTail:       "[]",
						Config:           &config,
						WandbConfig:      `{"t": 1}`,
					},
				},
			},
			expectError:   true,
			errorContains: "no log line count found",
		},
		{
			name: "Nil HistoryTail",
			response: ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:             "TestRun",
						HistoryLineCount: &historyLineCount,
						EventsLineCount:  &eventsLineCount,
						LogLineCount:     &logLineCount,
						EventsTail:       "[]",
						SummaryMetrics:   &summary,
						Config:           &config,
						WandbConfig:      `{"t": 1}`,
					},
				},
			},
			expectError:   true,
			errorContains: "no history tail found",
		},
		{
			name: "Nil SummaryMetrics",
			response: ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:             "TestRun",
						HistoryLineCount: &historyLineCount,
						EventsLineCount:  &eventsLineCount,
						LogLineCount:     &logLineCount,
						EventsTail:       "[]",
						HistoryTail:      &history,
						Config:           &config,
						WandbConfig:      `{"t": 1}`,
					},
				},
			},
			expectError:   true,
			errorContains: "no summary metrics found",
		},
		{
			name: "Nil Config",
			response: ResumeResponse{
				Model: Model{
					Bucket: Bucket{
						Name:             "TestRun",
						HistoryLineCount: &historyLineCount,
						EventsLineCount:  &eventsLineCount,
						LogLineCount:     &logLineCount,
						EventsTail:       "[]",
						HistoryTail:      &history,
						SummaryMetrics:   &summary,
						WandbConfig:      `{"t": 1}`,
					},
				},
			},
			expectError:   true,
			errorContains: "no config found",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockGQL := gqlmock.NewMockClient()

			jsonData, err := json.MarshalIndent(tc.response, "", "    ")
			assert.Nil(t, err, "Failed to marshal json data")

			mockGQL.StubMatchOnce(
				gqlmock.WithOpName("RunResumeStatus"),
				string(jsonData),
			)

			resumeState := runbranch.NewResumeBranch(
				context.Background(),
				mockGQL,
				"must") // Use "must" to ensure errors are returned

			params := &runbranch.RunParams{
				Entity:  "test-entity",
				Project: "test-project",
				RunID:   "test-run-id",
			}
			err = resumeState.UpdateForResume(params, runconfig.New())

			if tc.expectError {
				assert.NotNil(t, err, "GetUpdates should return an error")
				assert.Contains(t, err.Error(), tc.errorContains, "Error message should contain expected text")
			} else {
				assert.Nil(t, err, "GetUpdates should not return an error")
			}
		})
	}
}

func TestExtractRunStateAdjustsStartTime(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()

	historyLineCount := 5
	eventsLineCount := 10
	logLineCount := 15
	history := `["{\"_step\":4,\"_runtime\":100}"]`
	summary := `{"_runtime": 120, "_wandb": {"runtime": 130}}`
	config := `{}`
	eventsTail := `["{\"_runtime\":110}", "{\"_runtime\":120}"]`

	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "TestRun",
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				HistoryTail:      &history,
				SummaryMetrics:   &summary,
				Config:           &config,
				EventsTail:       eventsTail,
				WandbConfig:      `{"t": 1}`,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")

	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)

	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")

	params := &runbranch.RunParams{
		Entity:  "test-entity",
		Project: "test-project",
		RunID:   "test-run-id",
	}
	err = resumeState.UpdateForResume(params, runconfig.New())

	assert.Nil(t, err, "GetUpdates should not return an error")

	// Verify other fields are set correctly
	assert.Equal(t, int32(130), params.Runtime, "Runtime should be set to the maximum value")
	assert.True(t, params.Resumed, "Resumed flag should be set to true")
}

func TestResumedRunNotes(t *testing.T) {
	mockGQL := gqlmock.NewMockClient()

	notes := "test-notes"
	config := `{}`
	historyLineCount := 5
	eventsLineCount := 10
	logLineCount := 15
	summary := `{}`
	history := `[]`

	rr := ResumeResponse{
		Model: Model{
			Bucket: Bucket{
				Name:             "TestRun",
				EventsTail:       `[]`,
				WandbConfig:      `{"t": 1}`,
				Config:           &config,
				HistoryLineCount: &historyLineCount,
				EventsLineCount:  &eventsLineCount,
				LogLineCount:     &logLineCount,
				SummaryMetrics:   &summary,
				HistoryTail:      &history,
				Notes:            &notes,
			},
		},
	}

	jsonData, err := json.MarshalIndent(rr, "", "    ")
	assert.Nil(t, err, "Failed to marshal json data")

	mockGQL.StubMatchOnce(
		gqlmock.WithOpName("RunResumeStatus"),
		string(jsonData),
	)

	resumeState := runbranch.NewResumeBranch(
		context.Background(),
		mockGQL,
		"must")

	params := &runbranch.RunParams{
		Entity:  "test-entity",
		Project: "test-project",
		RunID:   "test-run-id",
	}

	err = resumeState.UpdateForResume(params, runconfig.New())
	assert.Nil(t, err, "GetUpdates should not return an error")

	assert.Equal(t, notes, params.Notes, "Notes should be set to the value from the response")
}
