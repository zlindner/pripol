import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import DataPractice from './dataPractice';

const Container = styled.div`
    width: 100vw;
    height: 100vh;
    padding: 40px;

    & > span {
        font-size: 18px;
    }
`;

const Grid = styled.div`
    display: grid;
    grid-template-columns: repeat(auto-fill, 300px);
    column-gap: 25px;
    row-gap: 25px;
    justify-content: center;
    align-items: center;
    position: relative;
    top: 50%;
    transform: translateY(-50%);
`;

interface ISegment {
    segment: string;
    data_practice: string;
}

type Props = {
    segments: ISegment[];
};

const dataPractices = [
    {
        name: 'First Party Collection/Use',
        about: 'How and why a service provider collects user information.',
        segments: [],
    },
    {
        name: 'Third Party Sharing/Collection',
        about: 'How user information may be shared with or collected by third parties.',
        segments: [],
    },
    {
        name: 'User Choice/Control',
        about: 'Choices and control options available to users.',
        segments: [],
    },
    {
        name: 'User Access, Edit, & Deletion',
        about: 'If and how user may access, edit, or delete their information.',
        segments: [],
    },
    {
        name: 'Data Retention',
        about: 'How long user information is stored.',
        segments: [],
    },
    {
        name: 'Data Security',
        about: 'How user information is protected.',
        segments: [],
    },
    {
        name: 'Policy Change',
        about: 'If and how users will be informed about changes to the privacy policy.',
        segments: [],
    },
    {
        name: 'Do Not Track',
        about: 'If and how Do Not Track signals for online tracking and advertising are honoured.',
        segments: [],
    },
    {
        name: 'International & Specific Audiences',
        about: 'Practices that pertain only to specific group of users.',
        segments: [],
    },
];

const Analysis = (props: Props) => {
    useEffect(() => {
        props.segments.forEach((segment) => {
            /*let name = segment.data_practice.split('_').

            dataPractices.find(d => d.name === segment.data_practice.split('_'))

            if (segment.data_practice === 'policy_change') {
                dataPractices.find(d => d.name === 'Policy Change')?.segments.
            }*/
        });

        console.log(props.segments);
    });

    return (
        <Container>
            <span>Detected data practices for {props.segments.length} segments.</span>

            <Grid>
                {dataPractices.map((d) => (
                    <DataPractice name={d.name} about={d.about} />
                ))}
            </Grid>
        </Container>
    );
};

export default Analysis;
